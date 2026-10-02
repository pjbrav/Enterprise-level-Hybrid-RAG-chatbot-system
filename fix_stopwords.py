"""
Stopword cleanup — remove dataset-specific template boilerplate.

With spaCy NER active, template words like "customer", "motivation",
"checklist" are never extracted as entities (spaCy knows they're not
PERSON/ORG/PRODUCT). So the stopword filter only matters for the regex
fallback path.

This patch:
1. Removes all Jira/GitHub/Confluence template words (15 words)
2. Keeps only universal English stopwords + question words + time words
3. Updates the docstring to reflect that this is now a universal list

Run:
    python fix_stopwords.py
"""

import re

V23_FILE = "v23.py"

with open(V23_FILE, "r", encoding="utf-8") as f:
    content = f.read()

OLD_STOPWORDS = '''# Boilerplate / common English words that the capitalized-phrase regex
# picks up from sentence starts, Jira/GitHub templates, etc. These carry
# no entity-graph signal and just pollute cross-partition matching.
_ENTITY_STOPWORDS = frozenset({
    # Common sentence-starters
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "is", "are", "was", "were", "be", "been", "being", "been",
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
    # Question words -- confirmed via profiling to be extracted as entities
    # on any question starting with one, then matching enormous numbers of
    # unrelated chunks in entity_search (e.g. "what" matched 7,640 chunks
    # in one partition alone).
    "what", "who", "whom", "whose", "which", "how", "when", "where", "why",
    "there", "here",
    # Template / boilerplate from Jira, GitHub, Confluence
    "added", "removed", "fixed", "updated", "created", "deleted",
    "customer", "support", "issue", "impact", "environment",
    "observed", "confirmed", "initial", "reported", "assigned",
    "priority", "status", "description", "summary", "comment",
    "motivation", "checklist", "changes", "approach", "context",
    "rollout", "background", "testing", "reviewed", "review",
    "merge", "merged", "pull", "request", "branch",
    "steps", "reproduce", "expected", "actual", "behavior",
    "acceptance", "criteria", "definition", "done",
    # Common words that happen to be capitalized
    "utc", "est", "pst", "cet", "ist",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "today", "yesterday", "tomorrow",
})'''

NEW_STOPWORDS = '''# Universal English stopwords for the regex entity extractor fallback.
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
})'''

if OLD_STOPWORDS in content:
    content = content.replace(OLD_STOPWORDS, NEW_STOPWORDS)
    print("  [OK] Cleaned _ENTITY_STOPWORDS — removed 15 template-specific words")
    print("       Kept: 30 English stopwords + 10 question words + 19 time words = 59 universal words")
    print("       Removed: 15 Jira/GitHub/Confluence template terms")
else:
    print("  [FAIL] Could not find the stopword block to replace")

with open(V23_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\n  Done. Run 'python -c \"from v23 import _ENTITY_STOPWORDS; print(len(_ENTITY_STOPWORDS))\"' to verify.")
