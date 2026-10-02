"""Read-only retrieval API for a hosted corpus.

One server serves ONE corpus. Members can only (1) join with a one-time code,
(2) ask for the top ranked chunks for a query, (3) leave. There is no ingest,
delete, list-documents or fetch-by-id endpoint, so nothing a member does can
alter the host's data, and they never receive more than `pool` chunks per
query. Answers are generated on the member's machine with the member's own
API key; the host never sees or pays for completions.

Transport is plain HTTP: use it on a trusted LAN or behind a tunnel.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from rag.corpora import CorpusManager
from rag.state import Membership, OtpError, RateLimited

API_VERSION = "1"
MAX_QUERY_CHARS = 2000
MAX_POOL = 50


class JoinRequest(BaseModel):
    otp: str = Field(min_length=1, max_length=32)
    display_name: str = Field(default="member", max_length=80)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    method: str = Field(default="hybrid", pattern="^(hybrid|standard)$")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)   # Hybrid slider: how much work this host does
    pool: int = Field(default=20, ge=1, le=MAX_POOL)


def create_host_app(manager: CorpusManager, corpus_id: str) -> FastAPI:
    state = manager.state
    corpus = manager.state.get_corpus(corpus_id)
    if corpus is None:
        raise KeyError(f"Unknown corpus: {corpus_id}")

    app = FastAPI(title="Hybrid RAG host", version=API_VERSION, docs_url=None, redoc_url=None, openapi_url=None)

    def current_member(authorization: Optional[str] = Header(default=None)) -> Membership:
        token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else ""
        member = state.verify_token(token, corpus_id=corpus_id)
        if member is None:
            raise HTTPException(status_code=401, detail="Invalid or revoked membership.")
        return member

    @app.get("/v1/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "corpus_name": corpus.name, "version": API_VERSION}

    @app.post("/v1/join")
    def join(body: JoinRequest, request: Request) -> Dict[str, Any]:
        ip = request.client.host if request.client else "unknown"
        try:
            token, member = state.redeem_otp(corpus_id, body.otp, body.display_name, ip)
        except RateLimited as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except OtpError as exc:
            raise HTTPException(status_code=401, detail=str(exc))
        fresh = state.get_corpus(corpus_id)
        return {"token": token, "corpus_name": fresh.name, "member_id": member.id,
                "partition_keywords": fresh.partition_keywords}

    @app.post("/v1/retrieve")
    def retrieve(body: RetrieveRequest, member: Membership = Depends(current_member)) -> Dict[str, Any]:
        _, _, engine = manager.get_engine(corpus_id)
        return engine.retrieve_for_remote(body.query, method=body.method, pool=body.pool, intensity=body.intensity)

    @app.post("/v1/leave")
    def leave(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else ""
        return {"ok": state.leave(token)}

    return app


def lan_ip() -> str:
    """Best-effort address other machines on the LAN can reach this one at."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))   # no packet is sent for a UDP connect
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


class HostServer:
    """Runs the host API in a background thread (uvicorn). The Gradio admin UI
    stays on 127.0.0.1; only this API is exposed to the network."""

    def __init__(self, manager: CorpusManager, corpus_id: str, host: str = "0.0.0.0", port: int = 8765):
        self.manager, self.corpus_id, self.host, self.port = manager, corpus_id, host, port
        self._server = None
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join_url(self) -> str:
        return f"http://{lan_ip()}:{self.port}"

    def start(self, timeout: float = 15.0) -> None:
        import uvicorn
        if self.running:
            return
        self.manager.state.set_kind(self.corpus_id, "hosted")
        config = uvicorn.Config(create_host_app(self.manager, self.corpus_id), host=self.host,
                                port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name=f"host-{self.port}", daemon=True)
        self._thread.start()
        deadline = time.time() + timeout
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError(f"Host server failed to start on port {self.port} (port in use?).")
            if time.time() > deadline:
                raise TimeoutError("Host server did not start in time.")
            time.sleep(0.05)

    def stop(self, remove_members: bool = False, timeout: float = 10.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
        self._server, self._thread = None, None
        if remove_members:
            self.manager.state.remove_all_members(self.corpus_id)
