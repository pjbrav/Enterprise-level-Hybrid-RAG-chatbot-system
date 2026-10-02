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


def available_provider_configs(preferred: str = "auto", key_override: Optional[str] = None) -> List[ProviderConfig]:
    def _resolve_key(name: str, env_var: str) -> str:
        if key_override and key_override.strip() and preferred == name:
            return key_override.strip()
        return os.getenv(env_var, "")

    configured = {
        "claude": ProviderConfig("claude", os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL), _resolve_key("claude", "ANTHROPIC_API_KEY")),
        "gemini": ProviderConfig("gemini", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), _resolve_key("gemini", "GEMINI_API_KEY")),
        "openai": ProviderConfig("openai", os.getenv("OPENAI_MODEL", "gpt-5.6-luna"), _resolve_key("openai", "OPENAI_API_KEY")),
        "compatible": ProviderConfig("compatible", os.getenv("COMPATIBLE_MODEL", ""), _resolve_key("compatible", "COMPATIBLE_API_KEY"), os.getenv("COMPATIBLE_BASE_URL", "")),
    }
    priority = ["claude", "gemini", "openai", "compatible"]
    if preferred != "auto":
        priority = [preferred] + [name for name in priority if name != preferred]
    return [configured[name] for name in priority if name in configured and configured[name].api_key and configured[name].model]


def choose_provider() -> str:
    choices = {"1": "auto", "2": "claude", "3": "gemini", "4": "openai", "5": "compatible"}
    print("\nGeneration provider: 1) Auto  2) Claude  3) Gemini  4) OpenAI  5) Compatible")
    return choices.get(input("Select provider [1]: ").strip() or "1", "auto")
