"""All the logic behind the Gradio UI, with no Gradio imports so it can be tested directly.

app.py is just layout + callbacks into AppController.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag.config import DATABASE_PATH
from rag.corpora import CorpusManager
from rag.hosting import HostServer
from rag.providers import ProviderConfig, available_provider_configs
from rag.remote import FederatedSearch, NoSourcesAvailable, RemoteCorpus, join_host
from rag.state import Corpus

ROOT = Path(__file__).resolve().parent.parent

PROVIDER_LABELS: Dict[str, str] = {
    "OpenAI": "openai",
    "Anthropic (Claude)": "claude",
    "Google Gemini": "gemini",
    "Ollama (local)": "ollama",
    "OpenAI-compatible": "compatible",
}
MODE_PRODUCTION = "Production (Hybrid only)"
MODE_TESTING = "Testing (Standard vs Hybrid)"
LARGE_DB_BYTES = 200 * 1024 * 1024   # don't COUNT(*) multi-million-row databases just to fill a table

# Which environment variables a child process needs for each provider.
_ENV_KEY = {"openai": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
            "compatible": "COMPATIBLE_API_KEY"}
_ENV_MODEL = {"openai": "OPENAI_MODEL", "claude": "ANTHROPIC_MODEL", "gemini": "GEMINI_MODEL",
              "ollama": "OLLAMA_MODEL", "compatible": "COMPATIBLE_MODEL"}


def server_keys_allowed() -> bool:
    """May a request without a typed key use API keys from this machine's environment/.env?
    Yes on a personal install (UI bound to localhost). No when the UI is reachable from
    the network, otherwise any visitor could spend the owner's key by leaving the box blank.
    UI_ALLOW_SERVER_KEYS=1/0 overrides."""
    explicit = os.getenv("UI_ALLOW_SERVER_KEYS", "").strip().lower()
    if explicit:
        return explicit in ("1", "true", "yes")
    return os.getenv("GRADIO_BIND", "127.0.0.1").strip() in ("127.0.0.1", "localhost", "::1")


@dataclass
class ProviderSettings:
    """What the user typed in the provider bar. Held per request; never written to
    disk or to os.environ."""
    provider: str = "openai"
    api_key: str = ""
    model: str = ""
    base_url: str = ""

    def configs(self) -> List[ProviderConfig]:
        """Only the selected provider: a failure is reported, never silently
        retried against another provider's key found in this machine's environment.
        Without a typed key, the server's own key is used only if server_keys_allowed()."""
        if self.provider != "ollama" and not self.api_key.strip() and not server_keys_allowed():
            return []
        return available_provider_configs(self.provider, key_override=self.api_key, model_override=self.model,
                                          base_url_override=self.base_url, only_preferred=True)


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


class AppController:
    def __init__(self, manager: Optional[CorpusManager] = None):
        self.manager = manager or CorpusManager()
        self.manager.register_exit_sweep()
        self.servers: Dict[str, HostServer] = {}

    # ------------------------------------------------------------------ sources
    def source_choices(self) -> List[Tuple[str, str]]:
        """(label, value) pairs for the search checklist: local corpora and joined hosts."""
        choices: List[Tuple[str, str]] = []
        for c in self.manager.list_corpora():
            tags = [t for t in (["read-only"] if c.read_only else []) + (["hosting"] if c.id in self.servers and self.servers[c.id].running else []) if t]
            suffix = f" ({', '.join(tags)})" if tags else ""
            choices.append((f"[mine] {c.name}{suffix}", f"local:{c.id}"))
        for j in self.manager.state.list_joins():
            state = " (access revoked)" if j.status == "revoked" else ""
            choices.append((f"[host] {j.corpus_name} @ {j.host_url}{state}", f"join:{j.id}"))
        return choices

    @staticmethod
    def _split_sources(sources: Sequence[str]) -> Tuple[List[str], List[int]]:
        local_ids = [s[len("local:"):] for s in sources if s.startswith("local:")]
        join_ids = [int(s[len("join:"):]) for s in sources if s.startswith("join:")]
        return local_ids, join_ids

    # ------------------------------------------------------------------ search
    def search(self, query: str, sources: Sequence[str], mode: str, intensity: float,
               settings: ProviderSettings) -> Dict[str, Any]:
        """Production: Hybrid only. Testing: Standard and Hybrid run independently and in
        parallel (latency ~ the slower one, not the sum); Standard is never copied into
        the Hybrid slot. Returns {'standard', 'hybrid', 'error', 'no_key'}."""
        query = (query or "").strip()
        if not query:
            return {"error": "Type a question first."}
        if not sources:
            return {"error": "Select at least one corpus to search (create one under **My library** or join a host)."}
        local_ids, join_ids = self._split_sources(sources)
        configs = settings.configs()
        fs = FederatedSearch(self.manager)

        def run(use_hybrid: bool) -> Dict[str, Any]:
            return fs.run(query, local_ids, join_ids, use_hybrid, configs, intensity=intensity)

        try:
            if mode == MODE_TESTING:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    standard_future, hybrid_future = pool.submit(run, False), pool.submit(run, True)
                    standard, hybrid = standard_future.result(), hybrid_future.result()
            else:
                standard, hybrid = None, run(True)
        except NoSourcesAvailable as exc:
            return {"error": "None of the selected corpora could be searched. " + " ".join(exc.warnings)}
        except Exception as exc:
            return {"error": f"Search failed: {type(exc).__name__}: {exc}"}
        return {"standard": standard, "hybrid": hybrid, "no_key": not configs}

    @staticmethod
    def token_savings(standard: Optional[Dict[str, Any]], hybrid: Dict[str, Any]) -> Optional[str]:
        if not standard or not standard.get("tokens_used"):
            return None
        s, h = standard["tokens_used"], hybrid["tokens_used"]
        pct = 100.0 * (s - h) / s
        return f"{pct:.1f}% fewer tokens than Standard ({s:,} -> {h:,})" if pct >= 0 else \
            f"{-pct:.1f}% MORE tokens than Standard ({s:,} -> {h:,})"

    @staticmethod
    def _source_lines(out: Dict[str, Any]) -> List[str]:
        lines = []
        for src in out.get("sources", []):
            chunk_id, file_name, partition = [p.strip() for p in src.split("|")]
            origin = chunk_id.split(":", 1)[0] if ":" in chunk_id else "mine"
            lines.append(f"- `[{origin}]` {file_name} - {partition.split(':')[-1]}  <sub>{chunk_id}</sub>")
        return lines

    def render_result(self, title: str, out: Optional[Dict[str, Any]], savings: Optional[str] = None,
                      no_key: bool = False) -> str:
        if out is None:
            return f"### {title}\n\n_Not run in this mode._"
        gen = out.get("generation", {})
        provider = gen.get("provider") or "local evidence extraction"
        rows = [
            ("Latency", f"{out['latency']:.1f} s"),
            ("Context tokens sent to the LLM", f"{out['tokens_used']:,}"),
        ]
        if savings:
            rows.append(("Token savings", savings))
        rows += [
            ("Retrieval confidence (heuristic)", f"{out['confidence_score']:.2f}"),
            ("Chunk relevance (lexical proxy, not accuracy)", f"{out['relevance_score']:.1f}%"),
            ("Answer relevance (lexical proxy, not accuracy)", f"{out['answer_relevance']:.1f}%"),
            ("Chunks used", str(out["chunks_retrieved"])),
            ("Answered by", provider),
        ]
        table = "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)
        parts = [f"### {title}"]
        for w in out.get("warnings", []):
            parts.append(f"> ⚠️ {w}")
        if no_key:
            hint = "" if server_keys_allowed() else " (this server does not use its own keys: type yours above)"
            parts.append("> ℹ️ No API key / model for the selected provider, so no LLM was called: "
                         f"the answer below is extracted straight from the retrieved evidence.{hint}")
        parts += [out["answer"], table]
        sources = self._source_lines(out)
        if sources:
            parts.append("**Sources**\n" + "\n".join(sources))
        return "\n\n".join(parts)

    def test_connection(self, settings: ProviderSettings) -> str:
        try:
            answer, info = self.manager.scratch_engine().test_provider_connection(
                settings.provider, settings.api_key, settings.model, settings.base_url)
        except Exception as exc:
            return f"❌ {type(exc).__name__}: {exc}"
        if info.get("provider"):
            return f"✅ {info['provider']} answered: {answer[:80]}"
        return f"❌ {info.get('notice') or 'No provider call was made.'}"

    # ------------------------------------------------------------------ library
    def add_documents(self, name: str, file_paths: Sequence[str], retain: bool) -> str:
        if not file_paths:
            return "Choose at least one file (pdf, txt, md, py, js, ipynb or a zip)."
        name = (name or "").strip() or "My documents"
        corpus = self.manager.state.find_by_name(name)
        if corpus is None:
            corpus = self.manager.create_corpus(name, kind="personal", retained=retain)
        elif corpus.read_only:
            return f"**{name}** is an attached read-only database; pick another name."
        try:
            summary = self.manager.ingest(corpus.id, list(file_paths))
        except Exception as exc:
            return f"❌ Ingestion failed: {type(exc).__name__}: {exc}"
        keywords = self.manager.state.get_corpus(corpus.id).partition_keywords
        kw = "\n".join(f"- `{pid}`: {', '.join(k.strip() for k in kws[:6])}" for pid, kws in keywords.items())
        kept = "kept after you close the app" if corpus.retained else "**one-shot**: removed when the app closes"
        return f"{summary.as_markdown()}\n\n**{corpus.name}** is {kept}.\n\n**Routing keywords** (edit under *Manage*):\n{kw or '_none yet_'}"

    def _counts(self, c: Corpus) -> Tuple[str, str, str]:
        try:
            if c.read_only and c.backend == "sqlite" and Path(c.location).stat().st_size > LARGE_DB_BYTES:
                return "large", "large", "large"
            db = self.manager._db_for(c)
            docs = db._query("SELECT COUNT(*) AS n FROM documents", fetch=True)[0]["n"]
            chunks = db._query("SELECT COUNT(*) AS n FROM chunks", fetch=True)[0]["n"]
            return str(docs), str(chunks), str(len(db.get_partition_ids()))
        except Exception:
            return "?", "?", "?"

    def corpus_rows(self) -> List[List[str]]:
        rows = []
        for c in self.manager.list_corpora():
            docs, chunks, parts = self._counts(c)
            hosting = c.id in self.servers and self.servers[c.id].running
            rows.append([c.name, "hosting" if hosting else c.kind, c.backend,
                         "read-only" if c.read_only else "writable",
                         "kept" if c.retained else "one-shot", docs, chunks, parts, c.id])
        return rows

    def corpus_choices(self, writable_only: bool = False) -> List[Tuple[str, str]]:
        return [(c.name, c.id) for c in self.manager.list_corpora() if not (writable_only and c.read_only)]

    def delete_corpus(self, corpus_id: str) -> str:
        c = self.manager.state.get_corpus(corpus_id or "")
        if c is None:
            return "Pick a corpus first."
        self.stop_hosting(corpus_id, remove_members=True)
        self.manager.delete_corpus(corpus_id)
        return f"Deleted **{c.name}**" + (" (the attached database file itself was not touched)." if c.read_only else ".")

    def attach_database(self, name: str, path: str) -> str:
        try:
            c = self.manager.attach_sqlite((name or "").strip() or Path(path).stem, path.strip().strip('"'), kind="personal")
        except Exception as exc:
            return f"❌ {exc}"
        return f"Attached **{c.name}** read-only. It is never copied or modified."

    def keywords_text(self, corpus_id: str) -> str:
        c = self.manager.state.get_corpus(corpus_id or "")
        if c is None:
            return ""
        return "\n".join(f"{pid}: {', '.join(kws)}" for pid, kws in sorted(c.partition_keywords.items()))

    def save_keywords(self, corpus_id: str, text: str) -> str:
        if not self.manager.state.get_corpus(corpus_id or ""):
            return "Pick a corpus first."
        parsed: Dict[str, List[str]] = {}
        for line in (text or "").splitlines():
            if ":" in line:
                pid, _, kws = line.partition(":")
                parsed[pid.strip()] = [k for k in (x.strip() for x in kws.split(",")) if k]
        self.manager.set_keywords(corpus_id, parsed)
        return f"Saved keywords for {len(parsed)} partition(s)."

    def regenerate_keywords(self, corpus_id: str) -> str:
        if not self.manager.state.get_corpus(corpus_id or ""):
            return "Pick a corpus first."
        try:
            self.manager.regenerate_keywords(corpus_id)
        except Exception as exc:
            return f"❌ {exc}"
        return "Regenerated keywords (manual edits were replaced)."

    # ------------------------------------------------------------------ host
    def start_hosting(self, corpus_id: str, port: int) -> str:
        c = self.manager.state.get_corpus(corpus_id or "")
        if c is None:
            return "Pick a corpus to host first."
        if c.id in self.servers and self.servers[c.id].running:
            return self.host_status(c.id)
        server = HostServer(self.manager, c.id, host=os.getenv("HOST_BIND", "0.0.0.0"), port=int(port))
        try:
            server.start()
        except Exception as exc:
            return f"❌ Could not start hosting on port {int(port)}: {exc}"
        self.servers[c.id] = server
        return self.host_status(c.id)

    def host_status(self, corpus_id: str) -> str:
        server = self.servers.get(corpus_id or "")
        if not server or not server.running:
            return "Not hosting. Click **Start hosting** to share this corpus."
        return (f"✅ Hosting. Give members this URL: **{server.join_url()}**\n\n"
                "Traffic is plain HTTP: use a trusted network or a tunnel. Windows may ask to allow Python "
                "through the firewall the first time. Members only ever receive the top few chunks for a "
                "question, never the files or the database.")

    def stop_hosting(self, corpus_id: str, remove_members: bool = False) -> str:
        server = self.servers.pop(corpus_id or "", None)
        if server:
            server.stop(remove_members=remove_members)
            return "Stopped hosting" + (" and removed all members." if remove_members else ". Members are kept.")
        if remove_members and corpus_id:
            self.manager.state.remove_all_members(corpus_id)
        return "Not hosting."

    def issue_otp(self, corpus_id: str, ttl_minutes: float, max_uses: int, rotate: bool = False) -> str:
        if not self.manager.state.get_corpus(corpus_id or ""):
            return "Pick a corpus first."
        ttl, uses = int(float(ttl_minutes) * 60), int(max_uses)
        code, expires = (self.manager.state.rotate_otp if rotate else self.manager.state.issue_otp)(corpus_id, ttl, uses)
        note = "" if corpus_id in self.servers and self.servers[corpus_id].running else "\n\n_You are not hosting yet: click **Start hosting**._"
        old = " All earlier codes no longer work; existing members stay." if rotate else ""
        return f"## Code: `{code}`\nValid until {_fmt_time(expires)} for up to {uses} join(s).{old}{note}"

    def members_rows(self, corpus_id: str) -> List[List[Any]]:
        return [[m.id, m.display_name, m.ip, m.joined_at, m.last_seen]
                for m in self.manager.state.list_members(corpus_id or "")]

    def kick(self, member_id: Any) -> str:
        try:
            ok = self.manager.state.kick(int(member_id))
        except (TypeError, ValueError):
            return "Enter the member's id from the table."
        return "Removed. Their access stopped immediately." if ok else "No such member."

    def security_banner(self, corpus_id: str) -> str:
        events = self.manager.state.security_events(corpus_id or "", limit=3)
        return "\n\n".join(f"> 🚨 {_fmt_time(e['ts'])}: {e['detail']}" for e in events)

    # ------------------------------------------------------------------ join
    def join(self, host_url: str, otp: str, display_name: str) -> str:
        try:
            record = join_host(self.manager, host_url, otp, display_name)
        except Exception as exc:
            return f"❌ {exc}"
        return f"✅ Joined **{record.corpus_name}** at {record.host_url}. Nothing was downloaded; tick it under **Search**."

    def joins_rows(self) -> List[List[Any]]:
        return [[j.id, j.corpus_name, j.host_url, j.display_name, j.status] for j in self.manager.state.list_joins()]

    def leave(self, join_id: Any) -> str:
        try:
            record = self.manager.state.get_join(int(join_id))
        except (TypeError, ValueError):
            record = None
        if record is None:
            return "Enter the id from the table."
        RemoteCorpus(record).leave()
        self.manager.state.remove_join(record.id)
        return f"Left **{record.corpus_name}**. Only the local membership token was stored, and it is gone."

    # ------------------------------------------------------------------ evaluate
    def run_eval(self, num_questions: int, settings: ProviderSettings) -> str:
        db_path = Path(DATABASE_PATH)
        if not db_path.is_absolute():
            db_path = ROOT / db_path
        if not db_path.exists():
            return f"Evaluation database not found at {db_path}. Attach/place hybrid_rag_v22.db under data/."
        if settings.provider == "ollama":
            pass  # no key needed
        elif not (settings.api_key or (server_keys_allowed() and os.getenv(_ENV_KEY.get(settings.provider, "")))):
            return "Enter an API key for the selected provider first."
        env = dict(os.environ)             # a COPY: the key goes to the child process only
        if settings.api_key and settings.provider in _ENV_KEY:
            env[_ENV_KEY[settings.provider]] = settings.api_key
        if settings.model:
            env[_ENV_MODEL[settings.provider]] = settings.model
        cmd = [sys.executable, str(ROOT / "scripts" / "batch_compare_v2.py"), "--force-hybrid",
               "--sample", str(int(num_questions)), "--output", "web_eval", "--provider", settings.provider,
               "--db", str(db_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(ROOT), env=env)
        except subprocess.TimeoutExpired:
            return "Evaluation timed out. Try fewer questions."
        text = result.stdout if result.returncode == 0 else (result.stderr or result.stdout)
        return text[-6000:]
