"""LLM provider configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from rag.config import DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------
@dataclass
class ProviderConfig:
    name: str
    model: str
    api_key: str
    base_url: Optional[str] = None


OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
PROVIDER_CHOICES = ("openai", "claude", "gemini", "ollama", "compatible")


def available_provider_configs(preferred: str = "auto", key_override: Optional[str] = None,
                               model_override: Optional[str] = None, base_url_override: Optional[str] = None,
                               only_preferred: bool = False) -> List[ProviderConfig]:
    """Provider configs, best first.

    A typed key / model / base URL applies to `preferred` only. With
    only_preferred=True nothing else is returned, so a failing provider is never
    silently replaced by another one billed to a key found in this machine's
    environment (what the UI wants); the default keeps the old fallback chain.
    Ollama needs no key and is only part of the 'auto' chain if OLLAMA_MODEL is set.
    """
    def _typed(value: Optional[str], name: str) -> str:
        return value.strip() if value and value.strip() and preferred == name else ""

    def _resolve_key(name: str, env_var: str) -> str:
        return _typed(key_override, name) or os.getenv(env_var, "")

    configured = {
        "claude": ProviderConfig("claude", _typed(model_override, "claude") or os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL, _resolve_key("claude", "ANTHROPIC_API_KEY")),
        "gemini": ProviderConfig("gemini", _typed(model_override, "gemini") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash", _resolve_key("gemini", "GEMINI_API_KEY")),
        "openai": ProviderConfig("openai", _typed(model_override, "openai") or os.getenv("OPENAI_MODEL") or "gpt-5.6-luna", _resolve_key("openai", "OPENAI_API_KEY")),
        "compatible": ProviderConfig("compatible", _typed(model_override, "compatible") or os.getenv("COMPATIBLE_MODEL", ""), _resolve_key("compatible", "COMPATIBLE_API_KEY"), _typed(base_url_override, "compatible") or os.getenv("COMPATIBLE_BASE_URL", "")),
        "ollama": ProviderConfig("ollama", _typed(model_override, "ollama") or os.getenv("OLLAMA_MODEL") or "llama3.1", "ollama", _typed(base_url_override, "ollama") or os.getenv("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE_URL),
    }
    priority = ["claude", "gemini", "openai", "compatible"] + (["ollama"] if os.getenv("OLLAMA_MODEL") else [])
    if preferred != "auto":
        priority = [preferred] + [name for name in priority if name != preferred]
        if only_preferred:
            priority = [preferred]
    return [configured[name] for name in priority if name in configured and configured[name].api_key and configured[name].model]


def choose_provider() -> str:
    choices = {"1": "auto", "2": "claude", "3": "gemini", "4": "openai", "5": "compatible", "6": "ollama"}
    print("\nGeneration provider: 1) Auto  2) Claude  3) Gemini  4) OpenAI  5) Compatible  6) Ollama (local)")
    return choices.get(input("Select provider [1]: ").strip() or "1", "auto")
