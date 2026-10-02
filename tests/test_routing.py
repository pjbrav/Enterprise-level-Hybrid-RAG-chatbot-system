"""Phase 4 tests: routing that works on any corpus, readable partition names,
heading-aware chunking, and parallel per-partition search."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from rag.config import CHUNK_WORDS
from rag.corpora import CorpusManager
from rag.ingest import DocumentProcessor
from rag.state import StateStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = sorted(str(p) for p in (ROOT / "samples" / "documents").rglob("*.md"))
PG_URL = os.getenv("TEST_DATABASE_URL", "").strip()


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


class TestChunking(unittest.TestCase):
    def legacy(self, text):
        ws = text.split()
        step = 195
        out = []
        for start in range(0, len(ws), step):
            out.append(ws[start:start + CHUNK_WORDS])
            if start + CHUNK_WORDS >= len(ws):
                break
        return out

    def test_text_without_headings_chunks_exactly_as_before(self):
        text = words(700)
        self.assertEqual(DocumentProcessor._chunk_word_lists(text), self.legacy(text))

    def test_chunks_start_at_headings(self):
        text = "\n\n".join(f"# Section {i}\n{words(150, f's{i}_')}" for i in range(3))
        chunks = DocumentProcessor._chunk_word_lists(text)
        self.assertEqual(len(chunks), 3)
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk[0:3], ["#", "Section", str(i)])
            self.assertFalse(any(w.startswith(f"s{(i + 1) % 3}_") for w in chunk), "sections must not bleed together")

    def test_small_sections_are_packed_together(self):
        text = "\n".join(f"## H{i}\n{words(30, f'p{i}_')}" for i in range(5))
        chunks = DocumentProcessor._chunk_word_lists(text)
        self.assertEqual(len(chunks), 1)
        self.assertLessEqual(len(chunks[0]), CHUNK_WORDS)

    def test_long_section_is_windowed_with_overlap(self):
        text = f"# Intro\n{words(40, 'a')}\n# Big\n{words(500, 'b')}"
        chunks = DocumentProcessor._chunk_word_lists(text)
        big = [c for c in chunks if any(w.startswith("b") for w in c)]
        self.assertGreaterEqual(len(big), 3)
        self.assertTrue(all(len(c) <= CHUNK_WORDS for c in chunks))
        self.assertEqual(big[0][-25:], big[1][:25], "windows of a long section keep the standard overlap")

    def test_every_word_is_kept(self):
        text = "\n".join(f"# S{i}\n{words(70 + 13 * i, f'q{i}_')}" for i in range(6))
        chunks = DocumentProcessor._chunk_word_lists(text)
        self.assertEqual({w for c in chunks for w in c}, set(text.split()))

    def test_pdf_still_loads_text(self):
        pdf = next((p for p in (ROOT / "samples" / "documents").rglob("*.pdf")), None)
        self.assertIsNotNone(pdf)
        self.assertGreater(len(DocumentProcessor._load_document(str(pdf)).split()), 20)


class TestRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.mgr = CorpusManager(state=StateStore(Path(cls._tmp.name) / "state.db"), database_url="",
                                corpora_dir=Path(cls._tmp.name) / "c", sweep_on_start=False)
        cls.corpus = cls.mgr.create_corpus("policies")
        cls.mgr.ingest(cls.corpus.id, SAMPLE_DOCS)

    @classmethod
    def tearDownClass(cls):
        try:
            cls._tmp.cleanup()
        except PermissionError:
            pass

    def keywords(self):
        return self.mgr.state.get_corpus(self.corpus.id).partition_keywords

    def engine(self):
        return self.mgr.get_engine(self.corpus.id)[2]

    def test_keywords_are_generated_for_every_partition(self):
        kws = self.keywords()
        db = self.mgr._db_for(self.corpus)
        self.assertEqual(set(kws), set(db.get_partition_ids()))
        self.assertIn("hr ", kws["partition_hr"], "short folder names get a trailing space")
        self.assertIn("finance", kws["partition_finance"])
        for pid, words_ in kws.items():
            self.assertGreaterEqual(len(words_), 2, pid)
            self.assertEqual(words_, [w.lower() for w in words_])

    def test_questions_route_to_the_right_partitions(self):
        eng = self.engine()
        self.assertIn("partition_hr", eng._get_active_partitions("What is the HR remote work policy?"))
        both = eng._get_active_partitions("Compare the finance authorization framework with the legal GDPR rules")
        self.assertIn("partition_finance", both)
        self.assertIn("partition_legal", both)
        self.assertTrue(eng._is_cross_domain_question("Compare the finance authorization framework with the legal GDPR rules"))

    def test_enterprise_defaults_do_not_leak_into_other_corpora(self):
        eng = self.engine()
        self.assertTrue(set(eng._partition_keywords()) <= set(self.mgr._db_for(self.corpus).get_partition_ids()))
        active = eng._get_active_partitions("email the ticket to the team in slack")
        self.assertFalse({"partition_gmail", "partition_jira", "partition_slack"} & set(active))
        self.assertFalse(eng._is_cross_domain_question("email the ticket to the team in slack"))

    def test_yaml_overrides_generated_keywords(self):
        eng = self.engine()
        yaml_path = Path(self._tmp.name) / "routing.yaml"
        yaml_path.write_text("partitions:\n  partition_hr:\n    keywords: [\"zebra\"]\n", encoding="utf-8")
        eng._load_routing_config(str(yaml_path))
        try:
            self.assertEqual(eng._partition_keywords()["partition_hr"], ["zebra"])
            self.assertIn("partition_hr", eng._get_active_partitions("tell me about the zebra rule"))
        finally:
            eng._yaml_keywords = {}
            eng._resolved_keywords = None

    def test_user_edited_keywords_survive_reingest(self):
        corpus = self.mgr.create_corpus("edited")
        self.mgr.ingest(corpus.id, SAMPLE_DOCS)
        kws = self.mgr.state.get_corpus(corpus.id).partition_keywords
        kws["partition_hr"] = ["wfh", "telework"]
        self.mgr.set_keywords(corpus.id, kws)
        self.mgr.ingest(corpus.id, SAMPLE_DOCS, force=True)
        self.assertEqual(self.mgr.state.get_corpus(corpus.id).partition_keywords["partition_hr"], ["wfh", "telework"])
        self.assertIn("partition_hr", self.mgr.get_engine(corpus.id)[2]._get_active_partitions("is wfh allowed"))
        regenerated = self.mgr.regenerate_keywords(corpus.id)
        self.assertNotEqual(regenerated["partition_hr"], ["wfh", "telework"])

    def test_parallel_partition_search_equals_sequential(self):
        eng = self.engine()
        q = "Compare the finance authorization framework with the legal GDPR supplier data processing and HR remote work"
        eng.PARALLEL_PARTITIONS = True
        parallel = eng._get_cross_domain_rankings(q, True)
        eng.PARALLEL_PARTITIONS = False
        try:
            sequential = eng._get_cross_domain_rankings(q, True)
        finally:
            eng.PARALLEL_PARTITIONS = True
        self.assertGreater(len(parallel), 0)
        self.assertGreaterEqual(len(eng._get_active_partitions(q)), 2)
        self.assertEqual(parallel, sequential)


class TestClusterNames(unittest.TestCase):
    def test_flat_upload_gets_readable_partition_names(self):
        topics = {
            "cooking": "recipe sauce simmer garlic butter oven roast pasta tomato basil kitchen flavour",
            "astronomy": "telescope galaxy nebula orbit planet comet stellar cosmos lunar eclipse asteroid",
        }
        with tempfile.TemporaryDirectory() as d:
            files = []
            for topic, vocab in topics.items():
                for i in range(3):
                    path = Path(d) / f"{topic}_{i}.txt"
                    path.write_text((vocab + " ") * 8 + f" note{i}", encoding="utf-8")
                    files.append(str(path))
            mgr = CorpusManager(state=StateStore(Path(d) / "state.db"), database_url="",
                                corpora_dir=Path(d) / "c", sweep_on_start=False)
            corpus = mgr.create_corpus("flat")
            mgr.ingest(corpus.id, files)
            parts = mgr._db_for(corpus).get_partition_ids()
            try:
                self.assertGreaterEqual(len(parts), 2, parts)
                self.assertFalse(any("cluster" in p for p in parts), parts)
                self.assertTrue(any(w in "".join(parts) for w in ("recipe", "sauce", "garlic", "pasta", "tomato", "butter")), parts)
                self.assertTrue(any(w in "".join(parts) for w in ("telescope", "galaxy", "nebula", "orbit", "planet", "stellar")), parts)
            finally:
                pass


@unittest.skipUnless(PG_URL, "set TEST_DATABASE_URL to run Postgres tests")
class TestPostgresRouting(unittest.TestCase):
    def test_keywords_generated_on_postgres(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CorpusManager(state=StateStore(Path(d) / "state.db"), database_url=PG_URL,
                                corpora_dir=Path(d) / "c", sweep_on_start=False)
            corpus = mgr.create_corpus("pg-routing")
            try:
                mgr.ingest(corpus.id, SAMPLE_DOCS)
                kws = mgr.state.get_corpus(corpus.id).partition_keywords
                self.assertIn("hr ", kws["partition_hr"])
                eng = mgr.get_engine(corpus.id)[2]
                self.assertIn("partition_hr", eng._get_active_partitions("What is the HR remote work policy?"))
            finally:
                mgr.delete_corpus(corpus.id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
