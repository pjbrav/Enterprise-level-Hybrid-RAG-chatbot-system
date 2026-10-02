"""Two-process smoke test of host / join, the way two machines would use it.

    python scripts/smoke_two_instances.py

A HOST process (its own data directory) ingests the sample documents, hosts them on a local port
and prints a one-time code. This process plays the MEMBER (a second data directory) and checks:

  1. joining needs no files and grows member storage by < 10 KB
  2. host-only search returns chunks tagged [host]; nothing from the host is written to disk
  3. the member's own documents are searched too, and mixed searches fuse both
  4. nothing the member does changes the host's corpus file (hash, size, mtime)
  5. Testing mode runs Standard and Hybrid independently
  6. after the host kicks the member, remote retrieval stops at once and local answers keep working
  7. repeated wrong codes get rate limited

No LLM is called (the member has no key and the UI policy refuses the machine's own keys), so it
costs nothing. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SAMPLES = sorted(str(p) for p in (ROOT / "samples" / "documents").rglob("*.md"))


def host_main(data_dir: str, port: int) -> None:
    """Child process: host the sample corpus, print {url, otp, corpus_file}, serve until told to quit."""
    from rag.corpora import CorpusManager
    from rag.hosting import HostServer
    from rag.state import StateStore

    d = Path(data_dir)
    mgr = CorpusManager(state=StateStore(d / "state.db"), database_url="", corpora_dir=d / "corpora",
                        sweep_on_start=False)
    corpus = mgr.create_corpus("Company policies", kind="hosted")
    mgr.ingest(corpus.id, SAMPLES)
    server = HostServer(mgr, corpus.id, host="127.0.0.1", port=port)
    server.start()
    code, _ = mgr.state.issue_otp(corpus.id)
    print(json.dumps({"url": f"http://127.0.0.1:{port}", "otp": code, "corpus_file": corpus.location}), flush=True)
    for line in sys.stdin:
        cmd = line.strip()
        if cmd.startswith("kick "):
            name = cmd[5:]
            for member in mgr.state.list_members(corpus.id):
                if member.display_name == name:
                    mgr.state.kick(member.id)
            print("kicked", flush=True)
        elif cmd == "quit":
            break
    server.stop()


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def fingerprint(path: str):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size, p.stat().st_mtime_ns


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["host", "member"], default="member")
    parser.add_argument("--data")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.role == "host":
        host_main(args.data, args.port)
        return 0

    # The member must never reach a real LLM: network-bound UI policy => no server keys.
    os.environ["GRADIO_BIND"] = "0.0.0.0"
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "COMPATIBLE_API_KEY", "OLLAMA_MODEL"):
        os.environ.pop(key, None)

    from rag.app_controller import MODE_PRODUCTION, MODE_TESTING, AppController, ProviderSettings
    from rag.corpora import CorpusManager
    from rag.state import StateStore

    host_dir, member_dir = Path(tempfile.mkdtemp(prefix="rag_host_")), Path(tempfile.mkdtemp(prefix="rag_member_"))
    port = free_port()
    env = {**os.environ, "PYTHONUTF8": "1"}
    host = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--role", "host",
                             "--data", str(host_dir), "--port", str(port)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, env=env, cwd=str(ROOT))
    lines: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in host.stdout], daemon=True).start()

    results = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        results.append((name, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
        return ok

    try:
        print("Starting the host process (ingesting sample documents)...", flush=True)
        info = None
        deadline = time.time() + 300
        while time.time() < deadline and info is None:
            try:
                line = lines.get(timeout=2)
                if line.startswith("{"):
                    info = json.loads(line)
            except queue.Empty:
                if host.poll() is not None:
                    break
        if info is None:
            print("FAIL: the host process did not come up")
            return 1
        print(f"Host ready at {info['url']}\n", flush=True)

        ctl = AppController(CorpusManager(state=StateStore(member_dir / "state.db"), database_url="",
                                          corpora_dir=member_dir / "corpora", sweep_on_start=False))
        no_key = ProviderSettings("openai")
        host_before = fingerprint(info["corpus_file"])
        size_before = dir_size(member_dir)

        # 1. join with no files
        msg = ctl.join(info["url"], info["otp"], "smoke-member")
        grown = dir_size(member_dir) - size_before
        check("member joins without any files", "Joined" in msg, msg[:60])
        check("joining grows member storage by < 10 KB", grown < 10 * 1024, f"{grown} bytes")
        join_id = ctl.manager.state.list_joins()[-1].id

        # 2. host-only search
        out = ctl.search("What is the remote work policy?", [f"join:{join_id}"], MODE_PRODUCTION, 0.5, no_key)
        hyb = out.get("hybrid") or {}
        check("host-only search returns [host] chunks", bool(hyb.get("chunk_ids")) and
              all(c.startswith("host:") for c in hyb["chunk_ids"]), str(out.get("error", ""))[:80])
        blobs = b"".join(p.read_bytes() for p in member_dir.rglob("*") if p.is_file())
        check("no host document text was written to the member's disk",
              all(n not in blobs for n in (b"HR-POL-030", b"Supplier Qualification", b"AeroVelo")))

        # 3. member's own overlay + mixed search
        notes = member_dir / "zeppelin.md"
        notes.write_text("# Zeppelin maintenance\n\nThe zeppelin hangar inspection checklist requires weekly helium "
                         "valve checks and a rigging audit signed by the airship engineer. Every inspection must record "
                         "the helium pressure, the condition of the envelope fabric and the tension of all rigging "
                         "cables before the zeppelin may leave the hangar again.\n", encoding="utf-8")
        ctl.add_documents("My notes", [str(notes)], retain=True)
        mine = ctl.manager.state.find_by_name("My notes")
        local_only = ctl.search("zeppelin helium valve checklist", [f"local:{mine.id}"], MODE_PRODUCTION, 0.5, no_key)
        check("member's own documents are searchable", local_only.get("hybrid", {}).get("chunks_retrieved", 0) > 0)
        mixed = ctl.search("zeppelin helium checklist and remote work policy", [f"local:{mine.id}", f"join:{join_id}"],
                           MODE_PRODUCTION, 0.5, no_key)
        origins = set((mixed.get("hybrid") or {}).get("origins", {}))
        check("a mixed search uses both corpora without warnings", origins <= {"host", "mine"} and
              not (mixed.get("hybrid") or {}).get("warnings") and bool(origins), str(origins))

        # 5. testing mode
        both = ctl.search("What is the remote work policy?", [f"join:{join_id}"], MODE_TESTING, 0.5, no_key)
        check("Testing mode runs Standard and Hybrid independently",
              bool(both.get("standard")) and bool(both.get("hybrid")) and both["standard"] is not both["hybrid"])

        # 4. host untouched by everything above
        check("the host's corpus file is unchanged by member activity", fingerprint(info["corpus_file"]) == host_before)

        # 6. kick
        host.stdin.write("kick smoke-member\n")
        host.stdin.flush()
        lines.get(timeout=30)
        after = ctl.search("zeppelin helium checklist and remote work policy", [f"local:{mine.id}", f"join:{join_id}"],
                           MODE_PRODUCTION, 0.5, no_key)
        h = after.get("hybrid") or {}
        check("after a kick, remote retrieval stops immediately", any("unavailable" in w for w in h.get("warnings", [])),
              "; ".join(h.get("warnings", []))[:70])
        check("...and local answers keep working", h.get("chunks_retrieved", 0) > 0 and
              all(c.startswith("mine:") for c in h.get("chunk_ids", [])))
        check("the join is marked revoked", ctl.manager.state.get_join(join_id).status == "revoked")

        # 7. rate limit (last: it blocks this address for a while)
        statuses = [ctl.join(info["url"], "000000", "attacker") for _ in range(6)]
        check("repeated wrong codes get rate limited", "Too many failed attempts" in statuses[-1], statuses[-1][:60])
        check("the host's corpus file is still unchanged", fingerprint(info["corpus_file"]) == host_before)
    finally:
        try:
            host.stdin.write("quit\n")
            host.stdin.flush()
            host.wait(timeout=20)
        except Exception:
            host.kill()
        for d in (host_dir, member_dir):
            shutil.rmtree(d, ignore_errors=True)

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
