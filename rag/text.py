"""Text cleaning, tokenisation, stopwords and entity extraction."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import List, Set

try:
    import tiktoken
except ImportError:
    tiktoken = None


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "for",
    "with", "on", "at", "from", "by", "in", "and", "or", "but", "as",
    "what", "why", "how", "when", "where", "who", "which", "does", "do",
    "compare", "comparison", "contrast", "explain", "describe", "analyze",
    "analyse", "evaluate", "between", "both", "versus", "than", "into",
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    if text and tiktoken is not None:
        try:
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            pass
    return math.ceil(len(text) / 4) if text else 0


def stem_token(token: str) -> str:
    token = token.lower().strip("-_")
    for suffix in ("izations", "ization", "ations", "ation", "ments", "ment", "ingly", "ing", "ies", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def query_terms(query: str) -> List[str]:
    return [
        stem_token(term) for term in re.findall(r"\b[\w-]+\b", query.lower())
        if len(term) > 2 and term not in STOPWORDS
    ]


# Universal English stopwords for the regex entity extractor fallback.
# With spaCy NER active, this list only affects the regex fallback path
# (used when spaCy is not installed). It contains only language-level
# stopwords — no dataset-specific template terms. Template boilerplate
# ("customer", "motivation", "checklist", etc.) was removed because:
#   1. spaCy doesn't extract these as entities (not PERSON/ORG/PRODUCT)
#   2. They are dataset-specific and don't generalize to new deployments
# Question words are kept because they pollute entity_search at scale
# regardless of dataset.
_ENTITY_STOPWORDS = frozenset({
    # Articles, pronouns, prepositions, conjunctions
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "done",
    "will", "would", "should", "could", "can", "may", "might", "must",
    "for", "from", "with", "about", "into", "onto", "over", "under",
    "and", "or", "but", "if", "then", "else", "so", "as", "at", "by",
    "in", "on", "to", "of", "up", "out", "off", "down", "away",
    "yes", "no", "not", "nor", "yet", "ok", "okay",
    "you", "your", "yours", "we", "our", "ours", "us",
    "i", "me", "my", "mine", "they", "them", "their", "theirs",
    "he", "she", "his", "her", "hers", "him",
    "please", "thanks", "thank", "hi", "hey", "hello",
    # Question words — universal: "what" matches thousands of chunks
    # in any corpus, not just this one
    "what", "who", "whom", "whose", "which", "how", "when", "where", "why",
    "there", "here",
    # Time words — universal abbreviations capitalized in timestamps
    "utc", "est", "pst", "cet", "ist",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "today", "yesterday", "tomorrow",
})


_SPACY_NLP = None


_SPACY_AVAILABLE = False


try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    pass


_SPACY_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "NORP", "FAC", "EVENT",
    "WORK_OF_ART", "LAW", "PRODUCT", "LOC",
}


def _get_spacy_nlp():
    """Lazily load spaCy model on first use (saves startup time)."""
    global _SPACY_NLP
    if _SPACY_NLP is None and _SPACY_AVAILABLE:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except Exception:
            try:
                _SPACY_NLP = spacy.load("en_core_web_md", disable=["parser", "lemmatizer"])
            except Exception:
                _SPACY_NLP = None
    return _SPACY_NLP


def _add_regex_ids(text: str, entities: Set[str]) -> None:
    """Extract structured IDs that spaCy misses."""
    # Ticket IDs: PROJ-123, ENG-4567
    for m in re.findall(r"\b([A-Z]{2,10}-\d{1,8})\b", text):
        entities.add(m.lower())
    # PR/issue references: #1234
    for m in re.findall(r"#(\d{2,8})\b", text):
        entities.add(f"#{m}")
    # Commit hashes: 7-40 hex chars
    for m in re.findall(r"\b([0-9a-f]{7,40})\b", text):
        if len(m) >= 7:
            entities.add(m)
    # Version numbers: v2.3.1, 2.0.0
    for m in re.findall(r"\b(v?\d+\.\d+\.\d+)\b", text):
        entities.add(m.lower())


def extract_entities(text: str) -> Set[str]:
    """Extract entities using spaCy NER + regex ID fallback.

    spaCy provides context-aware named entity recognition (PERSON, ORG,
    PRODUCT, etc.) that is far more accurate than regex for real names.
    A regex fallback catches structured IDs spaCy misses: ticket IDs,
    PR numbers, commit hashes.

    All entities are lowercased. If spaCy is not installed or the model
    is not downloaded, falls back to the original regex-only extractor.
    """
    entities = set()

    # --- spaCy NER path ---
    nlp = _get_spacy_nlp()
    if nlp is not None:
        truncated = text[:50000] if len(text) > 50000 else text
        doc = nlp(truncated)
        for ent in doc.ents:
            if ent.label_ in _SPACY_ENTITY_TYPES:
                name = ent.text.strip().lower()
                if len(name) >= 3 and name not in _ENTITY_STOPWORDS:
                    entities.add(name)
        _add_regex_ids(text, entities)
        return entities

    # --- Regex fallback (spaCy not available) ---
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*)*\b", text)
    for c in candidates:
        c = c.strip().lower()
        if len(c) >= 3 and c not in _ENTITY_STOPWORDS:
            entities.add(c)
    number_pattern = r"\b[\$€]?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*(?:psi|USD|EUR|bar|%|k|M)?\b"
    for num in re.findall(number_pattern, text):
        num = num.strip()
        if num and len(num) > 2:
            entities.add(num)
    id_pattern = r"\b\d{4,}\b"
    for id_val in re.findall(id_pattern, text):
        if len(id_val) == 4 and 1900 <= int(id_val) <= 2099:
            continue
        entities.add(id_val)
    _add_regex_ids(text, entities)
    return {e for e in entities if len(e) > 2}
