"""Phase 5 tests: the Hybrid intensity slider and the provider layer."""

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from fastapi.testclient import TestClient

from rag.corpora import CorpusManager
from rag.hosting import create_host_app
from rag.params import HybridParams
from rag.providers import OLLAMA_DEFAULT_BASE_URL, available_provider_configs
from rag.state import StateStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = sorted(str(p) for p in (ROOT / "samples" / "documents").rglob("*.md"))
MULTI = "Compare the finance authorization framework with the legal GDPR rules and the HR remote work policy"


class TestIntensityMapping(unittest.TestCase):
    BASE = HybridParams(30, 3, 0.40, 20)

    def test_midpoint_is_exactly_the_base(self):
        self.assertIs(HybridParams.for_intensity(0.5, self.BASE), self.BASE)

    def test_anchors(self):
        self.assertEqual(HybridParams.for_intensity(0.0, self.BASE), HybridParams(15, 1, 0.0, 10))
        self.assertEqual(HybridParams.for_intensity(1.0, self.BASE), HybridParams(80, 3, 0.65, 40))

    def test_clamped(self):
        self.assertEqual(HybridParams.for_intensity(-3, self.BASE), HybridParams.for_intensity(0, self.BASE))
        self.assertEqual(HybridParams.for_intensity(9, self.BASE), HybridParams.for_intensity(1, self.BASE))

    def test_monotonic_in_every_field(self):
        previous = None
        for i in range(0, 21):
            p = HybridParams.for_intensity(i / 20, self.BASE)
            if previous:
                self.assertGreaterEqual(p.lexical_limit, previous.lexical_limit)
                self.assertGreaterEqual(p.max_partitions, previous.max_partitions)
                self.assertGreaterEqual(p.last_resort_threshold, previous.last_resort_threshold)
                self.assertGreaterEqual(p.last_resort_pool, previous.last_resort_pool)
            previous = p

    def test_midpoint_follows_a_non_default_base(self):
        base = HybridParams(50, 4, 0.5, 25)
        self.assertIs(HybridParams.for_intensity(0.5, base), base)
        high = HybridParams.for_intensity(1.0, base)
        self.assertEqual((high.max_partitions, high.last_resort_threshold), (4, 0.65))
        self.assertGreater(high.lexical_limit, base.lexical_limit)


class TestSliderChangesWhatTheEngineDoes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        state = StateStore(Path(cls._tmp.name) / "state.db")
        cls.mgr = CorpusManager(state=state, database_url="", corpora_dir=Path(cls._tmp.name) / "c", sweep_on_start=False)
        cls.corpus = cls.mgr.create_corpus("policies", kind="hosted")
        cls.mgr.ingest(cls.corpus.id, SAMPLE_DOCS)
        cls.db, _, cls.engine = cls.mgr.get_engine(cls.corpus.id)

    @classmethod
    def tearDownClass(cls):
        try:
            cls._tmp.cleanup()
        except PermissionError:
            pass

    def spy_fts(self, params):
        calls = []
        real = self.db.fts_search

        def spy(terms, partition_id=None, limit=50):
            calls.append((partition_id, limit))
            return real(terms, partition_id=partition_id, limit=limit)

        with mock.patch.object(self.db, "fts_search", side_effect=spy):
            self.engine.retrieve(MULTI, True, params)
        return calls

    def test_default_params_equal_the_legacy_hard_coded_values(self):
        d = self.engine.default_params()
        self.assertEqual((d.lexical_limit, d.max_partitions, d.last_resort_threshold, d.last_resort_pool), (30, 3, 0.40, 20))
        self.assertEqual(self.engine.params_for_intensity(0.5), d)

    def test_midpoint_retrieval_is_identical_to_no_params(self):
        implicit = self.engine.retrieve(MULTI, True)
        explicit = self.engine.retrieve(MULTI, True, self.engine.params_for_intensity(0.5))
        self.assertEqual(implicit.ranked, explicit.ranked)
        self.assertEqual((implicit.is_cross_domain, implicit.threshold, implicit.max_chunks),
                         (explicit.is_cross_domain, explicit.threshold, explicit.max_chunks))

    def test_candidate_limit_follows_the_slider(self):
        low = {limit for _, limit in self.spy_fts(self.engine.params_for_intensity(0.0))}
        mid = {limit for _, limit in self.spy_fts(self.engine.params_for_intensity(0.5))}
        high = {limit for _, limit in self.spy_fts(self.engine.params_for_intensity(1.0))}
        self.assertEqual((low, mid, high), ({15}, {30}, {80}))

    def test_low_intensity_searches_fewer_partitions(self):
        searched = lambda calls: {pid for pid, _ in calls if pid}      # noqa: E731
        self.assertEqual(len(searched(self.spy_fts(self.engine.params_for_intensity(0.0)))), 1)
        self.assertGreaterEqual(len(searched(self.spy_fts(self.engine.params_for_intensity(0.5)))), 2)

    def test_cross_domain_is_only_demanded_when_several_partitions_are_searched(self):
        self.assertTrue(self.engine.retrieve(MULTI, True, self.engine.params_for_intensity(0.5)).is_cross_domain)
        self.assertFalse(self.engine.retrieve(MULTI, True, self.engine.params_for_intensity(0.0)).is_cross_domain)

    def test_last_resort_rerank_follows_the_slider(self):
        result = self.engine.retrieve(MULTI, True)
        with mock.patch.object(self.engine, "_last_resort_rerank", wraps=self.engine._last_resort_rerank) as spy:
            self.engine.select(MULTI, result, params=self.engine.params_for_intensity(0.0))
            self.assertEqual(spy.call_count, 0, "intensity 0 must never run the expensive rerank")
            self.engine.select(MULTI, result, params=HybridParams(30, 3, 2.0, 7))
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(spy.call_args.args[-1], 7, "the rerank must see last_resort_pool candidates")

    def test_standard_rag_ignores_the_slider(self):
        a = self.engine.retrieve(MULTI, False)
        b = self.engine.retrieve(MULTI, False, self.engine.params_for_intensity(0.0))
        self.assertEqual(a.ranked, b.ranked)
        self.assertIsNone(a.params)

    def test_host_api_does_the_work_the_member_asked_for(self):
        state = self.mgr.state
        client = TestClient(create_host_app(self.mgr, self.corpus.id), client=("198.51.100.7", 5000))
        code, _ = state.issue_otp(self.corpus.id)
        token = client.post("/v1/join", json={"otp": code, "display_name": "slider"}).json()["token"]
        counts = {}
        for intensity in (0.0, 1.0):
            r = client.post("/v1/retrieve", headers={"Authorization": f"Bearer {token}"},
                            json={"query": MULTI, "pool": 50, "intensity": intensity})
            self.assertEqual(r.status_code, 200)
            counts[intensity] = len(r.json()["chunks"])
        self.assertLess(counts[0.0], counts[1.0])


class _FakeOpenAICompatibleServer:
    """Minimal /v1/chat/completions endpoint standing in for Ollama."""

    def __init__(self):
        outer = self
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                payload = json.dumps({
                    "id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", ""),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "OLLAMA SAYS HELLO"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


class TestProviders(unittest.TestCase):
    CLEAN = {"ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": "", "OPENAI_API_KEY": "", "COMPATIBLE_API_KEY": "",
             "OLLAMA_MODEL": "", "OLLAMA_BASE_URL": "", "OPENAI_MODEL": "", "ANTHROPIC_MODEL": ""}

    def configs(self, env=None, **kw):
        with mock.patch.dict(os.environ, {}):
            for key in self.CLEAN:
                os.environ.pop(key, None)
            os.environ.update(env or {})
            return available_provider_configs(**kw)

    def test_ollama_is_a_first_class_keyless_provider(self):
        (cfg,) = self.configs(preferred="ollama")
        self.assertEqual((cfg.name, cfg.model, cfg.base_url), ("ollama", "llama3.1", OLLAMA_DEFAULT_BASE_URL))
        self.assertTrue(cfg.api_key)

    def test_ollama_model_and_url_can_be_overridden(self):
        (cfg,) = self.configs(preferred="ollama", model_override="mistral", base_url_override="http://box:11434/v1")
        self.assertEqual((cfg.model, cfg.base_url), ("mistral", "http://box:11434/v1"))

    def test_ollama_is_not_tried_in_auto_mode_unless_configured(self):
        self.assertEqual(self.configs(env={"OPENAI_API_KEY": "k"}, preferred="auto")[0].name, "openai")
        self.assertNotIn("ollama", [c.name for c in self.configs(env={"OPENAI_API_KEY": "k"}, preferred="auto")])
        self.assertIn("ollama", [c.name for c in self.configs(env={"OLLAMA_MODEL": "llama3.1"}, preferred="auto")])

    def test_typed_key_works_for_the_selected_provider_only(self):
        for provider in ("openai", "claude", "gemini"):
            cfgs = self.configs(env={"OPENAI_API_KEY": "env-key"}, preferred=provider, key_override="typed-key")
            self.assertEqual(cfgs[0].name, provider)
            self.assertEqual(cfgs[0].api_key, "typed-key")
            others = [c for c in cfgs if c.name != provider]
            self.assertTrue(all(c.api_key != "typed-key" for c in others))

    def test_typed_model_applies_to_the_selected_provider_only(self):
        cfgs = self.configs(env={"OPENAI_API_KEY": "a", "ANTHROPIC_API_KEY": "b"}, preferred="claude", model_override="my-model")
        self.assertEqual({c.name: c.model for c in cfgs}["claude"], "my-model")
        self.assertNotEqual({c.name: c.model for c in cfgs}["openai"], "my-model")

    def test_only_preferred_never_falls_back_to_keys_in_the_environment(self):
        cfgs = self.configs(env={"OPENAI_API_KEY": "owner-key"}, preferred="claude", key_override="theirs", only_preferred=True)
        self.assertEqual([c.name for c in cfgs], ["claude"])
        self.assertEqual(self.configs(env={"OPENAI_API_KEY": "owner-key"}, preferred="claude", only_preferred=True), [],
                         "no typed key and none in env for the selected provider -> nothing to call")

    def test_resolving_providers_never_writes_keys_into_the_environment(self):
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("OPENAI_API_KEY", None)
            available_provider_configs(preferred="openai", key_override="typed-key")
            self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_ollama_generation_round_trip(self):
        fake = _FakeOpenAICompatibleServer()
        tmp = tempfile.TemporaryDirectory()
        try:
            mgr = CorpusManager(state=StateStore(Path(tmp.name) / "s.db"), database_url="",
                                corpora_dir=Path(tmp.name) / "c", sweep_on_start=False)
            engine = mgr.scratch_engine()
            (cfg,) = self.configs(preferred="ollama", base_url_override=fake.url, model_override="llama3.1")
            answer, info = engine._generate_answer("What is up?", "some evidence", "test", [cfg])
            self.assertIn("OLLAMA SAYS HELLO", answer)
            self.assertEqual(info["provider"], "ollama")
            self.assertEqual(info["cost_usd"], 0.0)
            path, body = fake.requests[0]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(body["model"], "llama3.1")
            self.assertIn("max_tokens", body)
            self.assertNotIn("max_completion_tokens", body)
            self.assertIn("some evidence", body["messages"][0]["content"])
            self.assertNotIn("research papers", body["messages"][0]["content"])
        finally:
            fake.close()
            try:
                tmp.cleanup()
            except PermissionError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
