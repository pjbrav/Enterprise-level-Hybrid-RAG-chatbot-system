"""Phase 3 tests: a member joins a real host over HTTP, searches it, fuses it
with a local corpus, and degrades gracefully. Host and member use separate
data directories, exactly like two machines would."""

import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from rag.corpora import CorpusManager
from rag.hosting import HostServer
from rag.remote import (FederatedSearch, JoinFailed, NoSourcesAvailable, RemoteCorpus, fuse_results,
                        join_host, local_chunk_store)
from rag.state import StateStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = sorted(str(p) for p in (ROOT / "samples" / "documents").rglob("*.md"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def new_manager(directory: str) -> CorpusManager:
    return CorpusManager(state=StateStore(Path(directory) / "state.db"), database_url="",
                         corpora_dir=Path(directory) / "corpora", sweep_on_start=False)


class TestMember(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._host_dir = tempfile.TemporaryDirectory()
        cls._member_dir = tempfile.TemporaryDirectory()
        cls.host = new_manager(cls._host_dir.name)
        cls.member = new_manager(cls._member_dir.name)
        cls.shared = cls.host.create_corpus("shared", kind="hosted")
        cls.host.ingest(cls.shared.id, SAMPLE_DOCS)
        cls.server = HostServer(cls.host, cls.shared.id, host="127.0.0.1", port=free_port())
        cls.server.start()
        cls.url = f"http://127.0.0.1:{cls.server.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        for d in (cls._host_dir, cls._member_dir):
            try:
                d.cleanup()
            except PermissionError:
                pass

    def _join(self, name="member", manager=None):
        code, _ = self.host.state.issue_otp(self.shared.id)
        return join_host(manager or self.member, self.url, code, name)

    def _search(self, **kw):
        return FederatedSearch(self.member, remote_timeout=30).run(provider_configs=[], **kw)

    # -- joining ---------------------------------------------------------------
    def test_join_stores_only_a_token_never_host_data(self):
        member_dir = tempfile.TemporaryDirectory()
        try:
            mgr = new_manager(member_dir.name)
            record = self._join("tiny", manager=mgr)
            self.assertEqual(record.corpus_name, "shared")
            self.assertEqual(record.status, "active")
            files = [p for p in Path(member_dir.name).rglob("*") if p.is_file()]
            total = sum(p.stat().st_size for p in files)
            self.assertLess(total, 64 * 1024, f"member storage grew by {total} bytes: {files}")
            blob = b"".join(p.read_bytes() for p in files)
            for needle in (b"HR-POL-030", b"Supplier Qualification", b"Regional Governance"):
                self.assertNotIn(needle, blob, "host document content must never be stored by a member")
        finally:
            try:
                member_dir.cleanup()
            except PermissionError:
                pass

    def test_bad_code_is_refused_with_a_clear_message(self):
        self.host.state.issue_otp(self.shared.id)
        with self.assertRaisesRegex(JoinFailed, "invalid or has expired"):
            join_host(self.member, self.url, "000000", "x")

    def test_unreachable_host_is_reported(self):
        with self.assertRaisesRegex(JoinFailed, "Could not reach"):
            join_host(self.member, f"http://127.0.0.1:{free_port()}", "123456", "x", timeout=2)

    def test_brute_force_gets_rate_limited(self):
        guarded = self.host.create_corpus("guarded", kind="hosted")
        server = HostServer(self.host, guarded.id, host="127.0.0.1", port=free_port())
        server.start()
        try:
            self.host.state.issue_otp(guarded.id)
            messages = []
            for _ in range(6):
                try:
                    join_host(self.member, f"http://127.0.0.1:{server.port}", "000000", "x")
                except JoinFailed as exc:
                    messages.append(str(exc))
            self.assertIn("Too many failed attempts", messages[-1])
        finally:
            server.stop()

    # -- searching ----------------------------------------------------------------
    def test_host_only_search_matches_what_the_host_would_select(self):
        record = self._join("hostonly")
        q = "supplier qualification requirements"
        out = self._search(query=q, local_ids=[], join_ids=[record.id], use_hybrid=True)
        self.assertGreater(out["chunks_retrieved"], 0)
        self.assertTrue(all(c.startswith("host:") for c in out["chunk_ids"]))
        self.assertEqual(out["warnings"], [])
        _, _, host_engine = self.host.get_engine(self.shared.id)
        res = host_engine.retrieve(q, True)
        ids, conf, _ = host_engine.select(q, res)
        self.assertEqual([c[len("host:"):] for c in out["chunk_ids"]], ids)
        self.assertAlmostEqual(out["confidence_score"], conf, places=9)
        self.assertTrue(all("host" in s for s in out["sources"]) or out["sources"])

    def test_fusion_mixes_host_and_local_with_namespaced_ids_and_partitions(self):
        local = self.member.create_corpus("mine-notes")
        notes = Path(self._member_dir.name) / "zeppelin_notes.md"
        notes.write_text("# Zeppelin maintenance\n\nThe zeppelin hangar inspection checklist requires weekly "
                         "helium valve checks and a rigging audit signed by the airship engineer. Every "
                         "inspection must record the helium pressure, the condition of the envelope fabric, "
                         "and the tension of all rigging cables before the zeppelin may leave the hangar "
                         "again, and any failed valve check grounds the airship until a replacement is fitted.\n",
                         encoding="utf-8")
        self.member.ingest(local.id, [str(notes)])
        record = self._join("mixer")
        q = "zeppelin helium valve checklist and remote work policy"

        db, _, eng = self.member.get_engine(local.id)
        local_res = eng.retrieve(q, True)
        local_res.store = local_chunk_store(db, local_res)
        remote_res = RemoteCorpus(record, 30).retrieve(q, "hybrid", pool=20)
        fused = fuse_results(eng, [("mine", local_res), ("host", remote_res)], start=0.0)

        origins = {cid.split(":", 1)[0] for cid, _ in fused.ranked}
        self.assertEqual(origins, {"mine", "host"})
        parts = {fused.store.meta[cid].partition_id.split(":", 1)[0] for cid, _ in fused.ranked}
        self.assertEqual(parts, {"mine", "host"}, "partitions must be namespaced per origin")
        self.assertTrue(all(0.0 <= s <= 1.0 for _, s in fused.ranked))

        out = self._search(query=q, local_ids=[local.id], join_ids=[record.id], use_hybrid=True)
        self.assertEqual(out["warnings"], [])
        self.assertTrue(set(out["origins"]) <= {"mine", "host"})
        only_local = self._search(query="zeppelin helium valve checklist", local_ids=[local.id], join_ids=[], use_hybrid=True)
        self.assertIn("mine", only_local["origins"])

    def test_single_local_corpus_uses_the_unchanged_pipeline(self):
        local = self.member.create_corpus("plain")
        self.member.ingest(local.id, SAMPLE_DOCS)
        out = self._search(query="remote work policy", local_ids=[local.id], join_ids=[], use_hybrid=True)
        _, _, eng = self.member.get_engine(local.id)
        direct = eng._run_rag("remote work policy", "Hybrid RAG", True, [])
        self.assertEqual(out["chunk_ids"], direct["chunk_ids"])
        self.assertFalse(any(":" in c for c in out["chunk_ids"]))

    # -- failure handling -----------------------------------------------------------
    def test_kicked_member_gets_a_warning_and_still_gets_local_answers(self):
        local = self.member.create_corpus("fallback")
        self.member.ingest(local.id, SAMPLE_DOCS)
        record = self._join("to-be-kicked")
        ok = self._search(query="incident management", local_ids=[local.id], join_ids=[record.id], use_hybrid=True)
        self.assertEqual(ok["warnings"], [])
        mine = [m for m in self.host.state.list_members(self.shared.id) if m.display_name == "to-be-kicked"][0]
        self.host.state.kick(mine.id)
        out = self._search(query="incident management", local_ids=[local.id], join_ids=[record.id], use_hybrid=True)
        self.assertTrue(any("unavailable" in w for w in out["warnings"]), out["warnings"])
        self.assertGreater(out["chunks_retrieved"], 0)
        self.assertTrue(all(c.startswith("mine:") for c in out["chunk_ids"]))
        self.assertEqual(self.member.state.get_join(record.id).status, "revoked")
        with self.assertRaises(NoSourcesAvailable):
            self._search(query="incident management", local_ids=[], join_ids=[record.id], use_hybrid=True)

    def test_offline_host_degrades_to_local_only(self):
        local = self.member.create_corpus("offline-fallback")
        self.member.ingest(local.id, SAMPLE_DOCS)
        temp_host = HostServer(self.host, self.shared.id, host="127.0.0.1", port=free_port())
        temp_host.start()
        code, _ = self.host.state.issue_otp(self.shared.id)
        record = join_host(self.member, f"http://127.0.0.1:{temp_host.port}", code, "offline-test")
        temp_host.stop()
        out = self._search(query="remote work", local_ids=[local.id], join_ids=[record.id], use_hybrid=True)
        self.assertTrue(any("unreachable" in w for w in out["warnings"]), out["warnings"])
        self.assertGreater(out["chunks_retrieved"], 0)
        self.assertEqual(self.member.state.get_join(record.id).status, "active", "offline is not revocation")

    def test_leave_drops_the_membership_on_both_sides(self):
        record = self._join("leaver")
        RemoteCorpus(record).leave()
        self.member.state.remove_join(record.id)
        self.assertIsNone(self.member.state.get_join(record.id))
        self.assertFalse([m for m in self.host.state.list_members(self.shared.id) if m.display_name == "leaver"])

    def test_requires_at_least_one_source(self):
        with self.assertRaises(ValueError):
            self._search(query="x", local_ids=[], join_ids=[], use_hybrid=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
