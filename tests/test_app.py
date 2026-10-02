"""Phase 6 tests: the UI controller (everything behind the Gradio layout)."""

import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from rag.app_controller import MODE_PRODUCTION, MODE_TESTING, AppController, ProviderSettings
from rag.corpora import CorpusManager
from rag.state import StateStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = sorted(str(p) for p in (ROOT / "samples" / "documents").rglob("*.md"))
NO_KEY = ProviderSettings(provider="openai", api_key="")   # no LLM call: local evidence extraction


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def new_controller(directory: str) -> AppController:
    mgr = CorpusManager(state=StateStore(Path(directory) / "state.db"), database_url="",
                        corpora_dir=Path(directory) / "corpora", sweep_on_start=False)
    return AppController(mgr)


class TestController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # .env is loaded at import, so API keys may be in os.environ: tests must never reach a real LLM.
        cls._env = mock.patch.dict(os.environ, {"GRADIO_BIND": "127.0.0.1"})
        cls._env.start()
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "COMPATIBLE_API_KEY", "OLLAMA_MODEL",
                    "UI_ALLOW_SERVER_KEYS"):
            os.environ.pop(key, None)
        cls._dirs = [tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()]
        cls.host, cls.member = new_controller(cls._dirs[0].name), new_controller(cls._dirs[1].name)
        msg = cls.host.add_documents("Company policies", SAMPLE_DOCS, retain=True)
        assert "Routing keywords" in msg, msg
        cls.corpus = cls.host.manager.state.find_by_name("Company policies")

    @classmethod
    def tearDownClass(cls):
        cls._env.stop()
        for cid in list(cls.host.servers):
            cls.host.stop_hosting(cid)
        for d in cls._dirs:
            try:
                d.cleanup()
            except PermissionError:
                pass

    # ---- library ------------------------------------------------------------
    def test_library_lists_corpora_with_counts_and_choices(self):
        rows = self.host.corpus_rows()
        row = next(r for r in rows if r[0] == "Company policies")
        self.assertEqual(row[2:5], ["sqlite", "writable", "kept"])
        self.assertGreater(int(row[5]), 0)       # docs
        self.assertGreater(int(row[6]), 0)       # chunks
        self.assertIn(("[mine] Company policies", f"local:{self.corpus.id}"), self.host.source_choices())

    def test_one_shot_and_read_only_and_delete(self):
        c = new_controller(self._dirs[1].name + "_x") if False else self.member
        self.assertIn("one-shot", c.add_documents("temp", SAMPLE_DOCS[:2], retain=False))
        temp = c.manager.state.find_by_name("temp")
        self.assertFalse(temp.retained)
        self.assertIn("Attached", c.attach_database("ro", self.corpus.location))
        self.assertIn("read-only", c.add_documents("ro", SAMPLE_DOCS[:1], True))
        self.assertTrue(Path(self.corpus.location).exists())
        ro = c.manager.state.find_by_name("ro")
        self.assertIn("not touched", c.delete_corpus(ro.id))
        self.assertTrue(Path(self.corpus.location).exists())
        c.delete_corpus(temp.id)
        self.assertIsNone(c.manager.state.find_by_name("temp"))

    def test_keyword_editor_round_trip(self):
        text = self.host.keywords_text(self.corpus.id)
        self.assertIn("partition_hr:", text)
        self.host.save_keywords(self.corpus.id, text.replace("partition_hr:", "partition_hr: wfh,"))
        self.assertIn("wfh", self.host.keywords_text(self.corpus.id))
        self.host.regenerate_keywords(self.corpus.id)
        self.assertNotIn("wfh", self.host.keywords_text(self.corpus.id))

    # ---- search ---------------------------------------------------------------
    def test_input_validation(self):
        src = [f"local:{self.corpus.id}"]
        self.assertIn("Type a question", self.host.search("  ", src, MODE_PRODUCTION, 0.5, NO_KEY)["error"])
        self.assertIn("Select at least one", self.host.search("remote work", [], MODE_PRODUCTION, 0.5, NO_KEY)["error"])

    def test_production_runs_hybrid_only_testing_runs_both(self):
        src = [f"local:{self.corpus.id}"]
        prod = self.host.search("What is the remote work policy?", src, MODE_PRODUCTION, 0.5, NO_KEY)
        self.assertIsNone(prod["standard"])
        self.assertGreater(prod["hybrid"]["chunks_retrieved"], 0)
        self.assertTrue(prod["no_key"])
        test = self.host.search("What is the remote work policy?", src, MODE_TESTING, 0.5, NO_KEY)
        self.assertIsNotNone(test["standard"])
        self.assertIsNot(test["standard"], test["hybrid"])
        self.assertEqual(test["standard"]["retrieval_path"], "standard_rag")
        self.assertIn(test["hybrid"]["retrieval_path"], ("hybrid", "hybrid_last_resort"))

    def test_testing_mode_runs_the_two_pipelines_in_parallel(self):
        def slow(self_, query, local_ids, join_ids, use_hybrid, configs, **kw):
            time.sleep(0.6)
            return {"answer": "x", "tokens_used": 10, "latency": 0.6, "confidence_score": 0.5, "relevance_score": 1.0,
                    "answer_relevance": 1.0, "chunks_retrieved": 1, "sources": [], "generation": {}, "warnings": []}

        src = [f"local:{self.corpus.id}"]
        with mock.patch("rag.app_controller.FederatedSearch.run", slow):
            t = time.time()
            res = self.host.search("q", src, MODE_TESTING, 0.5, NO_KEY)
            elapsed = time.time() - t
        self.assertIsNotNone(res["standard"])
        self.assertLess(elapsed, 1.0, f"two 0.6s pipelines took {elapsed:.2f}s: they ran one after the other")

    def test_rendering_labels_relevance_as_a_proxy_and_shows_savings(self):
        std = {"tokens_used": 1000}
        hyb = {"answer": "Answer text", "latency": 1.5, "tokens_used": 800, "confidence_score": 0.8,
               "relevance_score": 40.0, "answer_relevance": 70.0, "chunks_retrieved": 2,
               "sources": ["host:abc|policy.md|partition_hr", "def|notes.md|mine:partition_x"],
               "generation": {"provider": "openai"}, "warnings": ["Shared corpus 'x' unavailable: boom."]}
        savings = AppController.token_savings(std, hyb)
        self.assertIn("20.0% fewer tokens", savings)
        self.assertIn("MORE tokens", AppController.token_savings({"tokens_used": 100}, {"tokens_used": 150}))
        md = self.host.render_result("Hybrid RAG", hyb, savings)
        self.assertIn("lexical proxy, not accuracy", md)
        self.assertNotIn("| Accuracy", md)
        self.assertIn("`[host]` policy.md", md)
        self.assertIn("`[mine]` notes.md", md)
        self.assertIn("⚠️ Shared corpus 'x' unavailable", md)
        self.assertIn("Not run in this mode", self.host.render_result("Standard RAG", None))

    def test_server_keys_are_only_used_on_localhost_installs(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "owner-key", "GRADIO_BIND": "127.0.0.1"}):
            self.assertEqual([c.api_key for c in ProviderSettings("openai").configs()], ["owner-key"])
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "owner-key", "GRADIO_BIND": "0.0.0.0"}):
            self.assertEqual(ProviderSettings("openai").configs(), [], "a network-reachable UI must not spend the owner's key")
            self.assertEqual([c.api_key for c in ProviderSettings("openai", "visitor-key").configs()], ["visitor-key"])
            self.assertEqual(len(ProviderSettings("ollama").configs()), 1, "Ollama needs no key")
            self.assertIn("does not use its own keys", self.host.render_result("Hybrid RAG", {
                "answer": "a", "latency": 0, "tokens_used": 0, "confidence_score": 0, "relevance_score": 0,
                "answer_relevance": 0, "chunks_retrieved": 0, "sources": [], "generation": {}}, no_key=True))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "owner-key", "GRADIO_BIND": "0.0.0.0", "UI_ALLOW_SERVER_KEYS": "1"}):
            self.assertEqual([c.api_key for c in ProviderSettings("openai").configs()], ["owner-key"])

    def test_typed_api_key_never_reaches_the_environment(self):
        secret = "sk-test-secret-123"
        src = [f"local:{self.corpus.id}"]
        settings = ProviderSettings("openai", secret, "gpt-x")
        self.assertEqual([c.api_key for c in settings.configs()], [secret])
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("OPENAI_API_KEY", None)
            self.host.search("remote work", src, MODE_PRODUCTION, 0.5, ProviderSettings("openai", secret, "gpt-x"))
            self.assertNotIn(secret, os.environ.values())
            self.assertNotIn("OPENAI_API_KEY", os.environ)

    # ---- host / join, end to end through the controller ----------------------------
    def test_host_issue_join_search_kick_flow(self):
        port = free_port()
        status = self.host.start_hosting(self.corpus.id, port)
        self.assertIn(f":{port}", status)
        self.assertIn("Hosting", status)
        try:
            self.assertIn("(hosting)", " ".join(label for label, _ in self.host.source_choices()))
            out = self.host.issue_otp(self.corpus.id, 30, 3)
            code = out.split("`")[1]
            self.assertRegex(code, r"^\d{6}$")
            self.assertIn("❌", self.member.join(f"http://127.0.0.1:{port}", "000000", "ana"))
            self.assertIn("Joined", self.member.join(f"http://127.0.0.1:{port}", code, "ana"))
            join = self.member.manager.state.list_joins()[-1]
            self.assertEqual(self.member.joins_rows()[-1][1], "Company policies")
            self.assertEqual([r[1] for r in self.host.members_rows(self.corpus.id)][-1], "ana")

            res = self.member.search("What is the remote work policy?", [f"join:{join.id}"], MODE_TESTING, 0.5, NO_KEY)
            self.assertTrue(all(c.startswith("host:") for c in res["hybrid"]["chunk_ids"]))
            self.assertTrue(all(c.startswith("host:") for c in res["standard"]["chunk_ids"]))
            self.assertIn("`[host]`", self.member.render_result("Hybrid RAG", res["hybrid"]))

            member_id = self.host.members_rows(self.corpus.id)[-1][0]
            self.assertIn("stopped immediately", self.host.kick(member_id))
            after = self.member.search("remote work", [f"join:{join.id}"], MODE_PRODUCTION, 0.5, NO_KEY)
            self.assertIn("None of the selected corpora", after["error"])
            self.assertEqual(self.member.joins_rows()[-1][-1], "revoked")
            self.assertIn("(access revoked)", " ".join(label for label, _ in self.member.source_choices()))
            self.assertIn("Left", self.member.leave(join.id))
        finally:
            self.assertIn("Stopped", self.host.stop_hosting(self.corpus.id))
        self.assertIn("Not hosting", self.host.host_status(self.corpus.id))

    def test_rotate_keeps_members_and_kills_old_codes(self):
        first = self.host.issue_otp(self.corpus.id, 30, 5).split("`")[1]
        second = self.host.issue_otp(self.corpus.id, 30, 5, rotate=True)
        self.assertIn("existing members stay", second)
        self.assertEqual(self.host.manager.state.active_otp_count(self.corpus.id), 1)
        self.assertNotEqual(first, second.split("`")[1]) if first == second.split("`")[1] else None

    # ---- evaluate ---------------------------------------------------------------------
    def test_eval_reports_missing_database_and_missing_key(self):
        settings = ProviderSettings("openai", "")
        with mock.patch("rag.app_controller.DATABASE_PATH", str(Path(self._dirs[0].name) / "none.db")):
            self.assertIn("not found", self.host.run_eval(5, settings))
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertIn("API key", self.host.run_eval(5, settings))

    def test_eval_passes_the_key_to_the_child_process_only(self):
        fake_db = Path(self._dirs[0].name) / "x.db"
        fake_db.write_bytes(b"")
        captured = {}

        def fake_run(cmd, **kw):
            captured.update(cmd=cmd, env=kw["env"])
            return mock.Mock(returncode=0, stdout="done", stderr="")

        with mock.patch("rag.app_controller.DATABASE_PATH", str(fake_db)), \
                mock.patch("rag.app_controller.subprocess.run", fake_run), mock.patch.dict(os.environ, {}):
            os.environ.pop("OPENAI_API_KEY", None)
            out = self.host.run_eval(10, ProviderSettings("openai", "sk-child-only", "gpt-x"))
            self.assertEqual(out, "done")
            self.assertEqual(captured["env"]["OPENAI_API_KEY"], "sk-child-only")
            self.assertEqual(captured["env"]["OPENAI_MODEL"], "gpt-x")
            self.assertNotIn("OPENAI_API_KEY", os.environ, "the parent environment must stay clean")
        self.assertIn("scripts", " ".join(captured["cmd"]))
        self.assertIn("--sample", captured["cmd"])


class TestUiBuilds(unittest.TestCase):
    def test_blocks_build_without_launching(self):
        import app
        with tempfile.TemporaryDirectory() as d:
            demo = app.build_app(new_controller(d))
            self.assertGreater(len(demo.blocks), 50)
            labels = {getattr(b, "label", None) for b in demo.blocks.values()}
            self.assertIn("Hybrid intensity (faster <- 0 ... 1 -> more thorough)", labels)
            self.assertIn("LLM provider", labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
