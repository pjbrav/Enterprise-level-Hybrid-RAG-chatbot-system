"""Phase 1 storage tests: per-corpus isolation, read-only attach, Postgres FTS.

SQLite tests need no setup. Postgres tests run only when TEST_DATABASE_URL is
set, e.g. postgresql://postgres:pw@localhost:5544/postgres (a throwaway server).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from rag.corpora import CorpusManager
from rag.db import DatabaseManager
from rag.state import StateStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = sorted(str(p) for p in (ROOT / "samples" / "documents").rglob("*.md"))
PG_URL = os.getenv("TEST_DATABASE_URL", "").strip()


def make_manager(tmp: str, database_url: str = "") -> CorpusManager:
    return CorpusManager(state=StateStore(Path(tmp) / "state.db"), database_url=database_url,
                         corpora_dir=Path(tmp) / "corpora", sweep_on_start=False)


class TestSqliteCorpora(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.mgr = make_manager(self.tmp)

    def tearDown(self):
        try:
            self._tmp.cleanup()
        except PermissionError:
            pass  # Windows: SQLite handle lingering; the temp dir is disposable

    def _ingested(self, name="docs", **kw):
        corpus = self.mgr.create_corpus(name, **kw)
        summary = self.mgr.ingest(corpus.id, SAMPLE_DOCS)
        self.assertGreater(summary.processed, 0)
        return corpus

    def test_sample_docs_exist(self):
        self.assertGreaterEqual(len(SAMPLE_DOCS), 5)

    def test_ingest_search_and_statistics(self):
        corpus = self._ingested()
        db = self.mgr._db_for(corpus)
        hits = db.fts_search(["remote", "work"], limit=10)
        self.assertGreater(len(hits), 0)
        self.assertGreater(sum(db.get_partition_counts().values()), 0)
        n_entities = db._query("SELECT COUNT(*) AS n FROM entity_frequency", fetch=True)[0]["n"]
        self.assertGreater(n_entities, 0, "entity_frequency must be rebuilt automatically after ingest")

    def test_reingest_leaves_no_stale_rows(self):
        corpus = self._ingested()
        db = self.mgr._db_for(corpus)
        before = db._query("SELECT COUNT(*) AS n FROM chunks", fetch=True)[0]["n"]
        self.mgr.ingest(corpus.id, SAMPLE_DOCS, force=True)
        after = db._query("SELECT COUNT(*) AS n FROM chunks", fetch=True)[0]["n"]
        fts = db._query("SELECT COUNT(*) AS n FROM chunks_fts", fetch=True)[0]["n"]
        self.assertEqual(before, after)
        self.assertEqual(after, fts, "FTS rows must match chunk rows after re-ingest")

    def test_corpora_are_isolated(self):
        a = self._ingested("a")
        b = self.mgr.create_corpus("b")
        self.assertGreater(len(self.mgr._db_for(a).fts_search(["remote"], limit=5)), 0)
        self.assertEqual(self.mgr._db_for(b).fts_search(["remote"], limit=5), [])
        self.assertNotEqual(a.location, b.location)

    def test_duplicate_name_rejected(self):
        self.mgr.create_corpus("same")
        with self.assertRaises(ValueError):
            self.mgr.create_corpus("same")

    def test_read_only_attach_never_modifies_file(self):
        src = self._ingested("src")
        path = Path(src.location)
        other = make_manager(os.path.join(self.tmp, "other"))
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        attached = other.attach_sqlite("shared", path)
        self.assertTrue(attached.read_only)
        db = other._db_for(attached)
        self.assertGreater(len(db.fts_search(["remote"], limit=5)), 0)
        db.save_comparison("q", {"hybrid_rag": {"answer": "x"}})     # bookkeeping: silently skipped
        db.record_usage("openai", "m", 10, 10)
        with self.assertRaises(PermissionError):
            db.save_document("x.md", "p", {}, {})
        with self.assertRaises(PermissionError):
            other.ingest(attached.id, SAMPLE_DOCS)
        self.assertEqual(before, (path.stat().st_size, path.stat().st_mtime_ns))

    def test_attach_rejects_non_rag_database(self):
        import sqlite3
        bad = Path(self.tmp) / "bad.db"
        sqlite3.connect(bad).execute("CREATE TABLE t (x)").connection.close()
        with self.assertRaises(ValueError):
            self.mgr.attach_sqlite("bad", bad)
        with self.assertRaises(FileNotFoundError):
            self.mgr.attach_sqlite("missing", Path(self.tmp) / "nope.db")

    def test_delete_removes_data_but_never_an_attached_file(self):
        corpus = self._ingested("gone")
        self.mgr.delete_corpus(corpus.id)
        self.assertFalse(Path(corpus.location).exists())
        self.assertIsNone(self.mgr.state.get_corpus(corpus.id))
        keep = self._ingested("keep")
        att = self.mgr.attach_sqlite("att", keep.location, kind="personal")
        self.mgr.delete_corpus(att.id)
        self.assertTrue(Path(keep.location).exists(), "deleting an attachment must not delete its file")

    def test_one_shot_corpora_are_swept(self):
        once = self.mgr.create_corpus("once", retained=False)
        keep = self.mgr.create_corpus("keep", retained=True)
        fresh = CorpusManager(state=self.mgr.state, database_url="",
                              corpora_dir=self.mgr.corpora_dir, sweep_on_start=True)
        self.assertIsNone(fresh.state.get_corpus(once.id))
        self.assertFalse(Path(once.location).exists())
        self.assertIsNotNone(fresh.state.get_corpus(keep.id))

    def test_engine_is_cached_and_retrieves(self):
        corpus = self._ingested("engine")
        first = self.mgr.get_engine(corpus.id)
        self.assertIs(first, self.mgr.get_engine(corpus.id))
        ranked = first[2]._get_cross_domain_rankings("remote work policy for employees", True)
        self.assertGreater(len(ranked), 0)
        self.mgr.invalidate(corpus.id)
        self.assertIsNot(first, self.mgr.get_engine(corpus.id))

    def test_pg_tsquery_builder(self):
        q = DatabaseManager._pg_tsquery(["remote", "deploy-bot", "ab", "x'; DROP TABLE chunks;--"])
        self.assertEqual(q, "remote:* | deploy:* | bot | ab | x | drop:* | table:* | chunks:*")
        self.assertNotIn(";", q)
        self.assertEqual(DatabaseManager._pg_tsquery(["!!!"]), "")


@unittest.skipUnless(PG_URL, "set TEST_DATABASE_URL to run Postgres tests")
class TestPostgresCorpora(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mgr = make_manager(self._tmp.name, database_url=PG_URL)
        self.corpus = self.mgr.create_corpus("pg-docs")
        self.mgr.ingest(self.corpus.id, SAMPLE_DOCS)
        self.db = self.mgr._db_for(self.corpus)

    def tearDown(self):
        try:
            self.mgr.delete_corpus(self.corpus.id)
        finally:
            try:
                self._tmp.cleanup()
            except PermissionError:
                pass

    def test_no_pragma_error_and_isolated_schema(self):
        self.assertTrue(self.corpus.location.startswith("corpus_"))
        self.assertGreater(sum(self.db.get_partition_counts().values()), 0)

    def test_fts_uses_or_semantics(self):
        both = self.db.fts_search(["remote", "zzzunmatchableterm"], limit=10)
        only = self.db.fts_search(["remote"], limit=10)
        self.assertGreater(len(both), 0, "an absent term must not eliminate matches (OR, not AND)")
        self.assertEqual({c for c, _ in both}, {c for c, _ in only})

    def test_statistics_tables_populated_and_capping_works(self):
        n_ent = self.db._query("SELECT COUNT(*) AS n FROM entity_frequency", fetch=True)[0]["n"]
        n_term = self.db._query("SELECT COUNT(*) AS n FROM term_df", fetch=True)[0]["n"]
        self.assertGreater(n_ent, 0)
        self.assertGreater(n_term, 0)
        many = ["remote", "work", "policy", "employee", "supplier", "data", "incident", "governance"]
        capped = self.db._cap_fts_terms(many)
        self.assertEqual(len(capped), self.db.MAX_FTS_QUERY_TERMS)

    def test_delete_drops_schema(self):
        corpus = self.mgr.create_corpus("pg-temp")
        self.mgr.delete_corpus(corpus.id)
        rows = DatabaseManager(database_url=PG_URL, schema=self.corpus.location)._query(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name = ?",
            (corpus.location,), fetch=True)
        self.assertEqual(rows, [])

    def test_engine_hybrid_and_standard_retrieval_on_postgres(self):
        _, _, engine = self.mgr.get_engine(self.corpus.id)
        hybrid = engine._get_cross_domain_rankings("remote work policy for employees", True)
        standard = engine._rank_candidates("remote work policy for employees", None, False)
        self.assertGreater(len(hybrid), 0)
        self.assertGreater(len(standard), 0)

    def test_two_corpora_do_not_see_each_other(self):
        other = self.mgr.create_corpus("pg-empty")
        try:
            self.assertEqual(self.mgr._db_for(other).fts_search(["remote"], limit=5), [])
        finally:
            self.mgr.delete_corpus(other.id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
