"""Phase 2 tests: OTP / membership rules and the read-only host API."""

import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import httpx
from fastapi.testclient import TestClient

from rag import state as state_mod
from rag.corpora import CorpusManager
from rag.hosting import HostServer, create_host_app
from rag.retrieval import wire_to_result
from rag.state import OtpError, RateLimited, StateStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = sorted(str(p) for p in (ROOT / "samples" / "documents").rglob("*.md"))


class TestOtpAndMembership(unittest.TestCase):
    """Pure state-store rules; no engine needed."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.st = StateStore(Path(self._tmp.name) / "state.db")
        self.cid = "corpus1"

    def tearDown(self):
        try:
            self._tmp.cleanup()
        except PermissionError:
            pass

    def test_code_is_six_digits_and_stored_hashed(self):
        code, expires = self.st.issue_otp(self.cid)
        self.assertRegex(code, r"^\d{6}$")
        self.assertGreater(expires, time.time())
        import sqlite3
        raw = sqlite3.connect(self.st.path).execute("SELECT code_hash, salt FROM otp_codes").fetchone()
        self.assertNotIn(code, raw[0] + raw[1])

    def test_redeem_returns_token_once_and_token_verifies(self):
        code, _ = self.st.issue_otp(self.cid)
        token, member = self.st.redeem_otp(self.cid, code, "Ana", "10.0.0.5")
        self.assertEqual(member.display_name, "Ana")
        self.assertEqual(self.st.verify_token(token, self.cid).id, member.id)
        self.assertIsNone(self.st.verify_token(token, "some-other-corpus"))
        import sqlite3
        stored = sqlite3.connect(self.st.path).execute("SELECT token_hash FROM memberships").fetchone()[0]
        self.assertNotEqual(stored, token)

    def test_wrong_expired_and_used_up_codes_are_rejected(self):
        with self.assertRaises(OtpError):
            self.st.redeem_otp(self.cid, "000000", "x", "1.1.1.1")
        expired, _ = self.st.issue_otp(self.cid, ttl_seconds=-5)
        with self.assertRaises(OtpError):
            self.st.redeem_otp(self.cid, expired, "x", "1.1.1.2")
        once, _ = self.st.issue_otp(self.cid, max_uses=1)
        self.st.redeem_otp(self.cid, once, "a", "2.2.2.2")
        with self.assertRaises(OtpError):
            self.st.redeem_otp(self.cid, once, "b", "2.2.2.3")

    def test_rate_limit_blocks_even_a_correct_code(self):
        code, _ = self.st.issue_otp(self.cid)
        for _ in range(state_mod.MAX_FAILED_JOINS_PER_IP):
            with self.assertRaises(OtpError):
                self.st.redeem_otp(self.cid, "999999" if code != "999999" else "888888", "x", "9.9.9.9")
        with self.assertRaises(RateLimited):
            self.st.redeem_otp(self.cid, code, "x", "9.9.9.9")
        token, _ = self.st.redeem_otp(self.cid, code, "ok", "8.8.8.8")   # another address is unaffected
        self.assertTrue(token)

    def test_burst_of_failures_revokes_all_codes_and_records_event(self):
        code, _ = self.st.issue_otp(self.cid)
        for i in range(state_mod.AUTO_REVOKE_FAILURES):
            with self.assertRaises(OtpError):
                self.st.redeem_otp(self.cid, "999999" if code != "999999" else "888888", "x", f"7.7.{i // 250}.{i % 250}")
        self.assertEqual(self.st.active_otp_count(self.cid), 0)
        with self.assertRaises(OtpError):
            self.st.redeem_otp(self.cid, code, "late", "6.6.6.6")
        events = self.st.security_events(self.cid)
        self.assertEqual(events[0]["kind"], "otp_auto_revoked")

    def test_rotate_kills_old_codes_but_keeps_members(self):
        old, _ = self.st.issue_otp(self.cid)
        token, _ = self.st.redeem_otp(self.cid, old, "stays", "3.3.3.3")
        new, _ = self.st.rotate_otp(self.cid)
        with self.assertRaises(OtpError):
            self.st.redeem_otp(self.cid, old, "x", "3.3.3.4")
        self.assertIsNotNone(self.st.verify_token(token, self.cid))
        self.assertTrue(self.st.redeem_otp(self.cid, new, "new", "3.3.3.5")[0])

    def test_kick_and_leave_invalidate_tokens(self):
        code, _ = self.st.issue_otp(self.cid)
        t1, m1 = self.st.redeem_otp(self.cid, code, "a", "4.4.4.4")
        t2, _ = self.st.redeem_otp(self.cid, code, "b", "4.4.4.5")
        self.assertTrue(self.st.kick(m1.id))
        self.assertIsNone(self.st.verify_token(t1, self.cid))
        self.assertTrue(self.st.leave(t2))
        self.assertIsNone(self.st.verify_token(t2, self.cid))
        self.assertEqual(self.st.list_members(self.cid), [])


class TestHostApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        state = StateStore(Path(cls._tmp.name) / "state.db")
        cls.mgr = CorpusManager(state=state, database_url="", corpora_dir=Path(cls._tmp.name) / "c",
                                sweep_on_start=False)
        cls.corpus = cls.mgr.create_corpus("shared", kind="hosted")
        cls.mgr.ingest(cls.corpus.id, SAMPLE_DOCS)
        cls.client = TestClient(create_host_app(cls.mgr, cls.corpus.id))
        cls.state = state

    @classmethod
    def tearDownClass(cls):
        try:
            cls._tmp.cleanup()
        except PermissionError:
            pass

    def _join(self, name="tester"):
        code, _ = self.state.issue_otp(self.corpus.id)
        r = self.client.post("/v1/join", json={"otp": code, "display_name": name})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def auth(self, token):
        return {"Authorization": f"Bearer {token}"}

    def test_health_needs_no_auth(self):
        r = self.client.get("/v1/health")
        self.assertEqual(r.json()["corpus_name"], "shared")

    def test_retrieve_requires_valid_token(self):
        body = {"query": "remote work policy"}
        self.assertEqual(self.client.post("/v1/retrieve", json=body).status_code, 401)
        self.assertEqual(self.client.post("/v1/retrieve", json=body, headers=self.auth("nope")).status_code, 401)

    def test_join_then_retrieve_returns_bounded_ranked_chunks(self):
        joined = self._join()
        self.assertEqual(joined["corpus_name"], "shared")
        r = self.client.post("/v1/retrieve", headers=self.auth(joined["token"]),
                             json={"query": "remote work policy for employees", "pool": 5})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertGreater(len(data["chunks"]), 0)
        self.assertLessEqual(len(data["chunks"]), 5)
        for c in data["chunks"]:
            self.assertTrue(c["text"])
            self.assertNotIn("/", c["file_name"])
            self.assertNotIn("\\", c["file_name"], "host file paths must not leak")
        scores = [c["score"] for c in data["chunks"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_wire_result_matches_local_engine_ranking(self):
        joined = self._join()
        q = "supplier qualification requirements"
        data = self.client.post("/v1/retrieve", headers=self.auth(joined["token"]), json={"query": q, "pool": 50}).json()
        _, _, engine = self.mgr.get_engine(self.corpus.id)
        local = engine.retrieve(q, use_hybrid=True)
        wire = wire_to_result(data)
        self.assertEqual([c for c, _ in wire.ranked], [c for c, _ in local.ranked[:50]])
        ids, conf, _ = engine.select(q, wire)
        local_ids, local_conf, _ = engine.select(q, local)
        self.assertEqual((ids, conf), (local_ids, local_conf))

    def test_kicked_member_loses_access_immediately(self):
        joined = self._join("soon-kicked")
        hdr = self.auth(joined["token"])
        self.assertEqual(self.client.post("/v1/retrieve", headers=hdr, json={"query": "incident"}).status_code, 200)
        self.state.kick(joined["member_id"])
        self.assertEqual(self.client.post("/v1/retrieve", headers=hdr, json={"query": "incident"}).status_code, 401)

    def test_leave_revokes_token(self):
        joined = self._join()
        hdr = self.auth(joined["token"])
        self.assertTrue(self.client.post("/v1/leave", headers=hdr).json()["ok"])
        self.assertEqual(self.client.post("/v1/retrieve", headers=hdr, json={"query": "x"}).status_code, 401)

    def test_bad_code_401_then_rate_limited_429(self):
        self.state.issue_otp(self.corpus.id)
        # a dedicated client address: limits are per IP, and the shared client must stay usable
        attacker = TestClient(create_host_app(self.mgr, self.corpus.id), client=("203.0.113.9", 4444))
        statuses = [attacker.post("/v1/join", json={"otp": "123456", "display_name": "x"}).status_code
                    for _ in range(state_mod.MAX_FAILED_JOINS_PER_IP + 1)]
        self.assertEqual(statuses[:-1], [401] * state_mod.MAX_FAILED_JOINS_PER_IP)
        self.assertEqual(statuses[-1], 429)

    def test_request_validation(self):
        joined = self._join()
        hdr = self.auth(joined["token"])
        for bad in ({"query": ""}, {"query": "x" * 2001}, {"query": "ok", "pool": 51},
                    {"query": "ok", "method": "evil"}, {"query": "ok", "intensity": 2}):
            self.assertEqual(self.client.post("/v1/retrieve", headers=hdr, json=bad).status_code, 422, bad)

    def test_there_is_no_admin_surface(self):
        joined = self._join()
        hdr = self.auth(joined["token"])
        for method, path in (("post", "/v1/ingest"), ("delete", "/v1/corpus"), ("get", "/v1/documents"),
                             ("get", "/v1/chunks/abc"), ("get", "/docs"), ("get", "/openapi.json")):
            self.assertEqual(getattr(self.client, method)(path, headers=hdr).status_code, 404, path)

    def test_real_server_start_serve_stop(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = HostServer(self.mgr, self.corpus.id, host="127.0.0.1", port=port)
        server.start()
        try:
            self.assertTrue(server.running)
            self.assertEqual(httpx.get(f"http://127.0.0.1:{port}/v1/health", timeout=5).status_code, 200)
        finally:
            server.stop()
        self.assertFalse(server.running)
        with self.assertRaises(httpx.TransportError):
            httpx.get(f"http://127.0.0.1:{port}/v1/health", timeout=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
