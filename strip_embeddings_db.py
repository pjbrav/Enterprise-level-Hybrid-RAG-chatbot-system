"""
Build hybrid_rag_v22_stripped.db — the FULL database (all 9 partitions,
all chunks, all entities) with only the embedding column emptied.

Why this is safe: the SQLite path in v24.py never reads the `embedding`
column back. Runtime embeddings (when that tier runs) are computed in
memory from chunk text. Verified: no `SELECT embedding` exists anywhere
in the code. Retrieval, answers, relevance scores, and latency are all
unchanged.

Disk needed: ~32GB free (16GB for the copy + ~16GB for VACUUM's rebuild).
Time: 15-40 minutes depending on disk speed.

Usage:  python strip_embeddings_db.py
"""

import os
import shutil
import sqlite3
import time

SOURCE = "hybrid_rag_v22.db"
STRIPPED = "hybrid_rag_v22_stripped.db"


def size_gb(path: str) -> float:
    return os.path.getsize(path) / 1e9


def free_gb(path: str = ".") -> float:
    return shutil.disk_usage(path).free / 1e9


def one(cur, sql):
    return cur.execute(sql).fetchone()[0]


# ---- 0. Preflight checks ----------------------------------------------------
print(f"Source: {SOURCE} ({size_gb(SOURCE):.2f} GB)")

if not os.path.exists(SOURCE):
    raise SystemExit(f"ERROR: {SOURCE} not found in current directory.")

need = size_gb(SOURCE) * 2
have = free_gb()
print(f"Free disk space: {have:.1f} GB (need ~{need:.0f} GB for copy + VACUUM)")
if have < need:
    raise SystemExit(
        f"ERROR: not enough free disk space. Free up ~{need:.0f} GB and re-run.\n"
        f"(The copy needs {size_gb(SOURCE):.0f} GB, and VACUUM's rebuild needs\n"
        f"roughly the post-strip size again in temp space.)"
    )

# ---- 1. Copy ------------------------------------------------------------------
print(f"\nCopying {SOURCE} -> {STRIPPED} ...")
if os.path.exists(STRIPPED):
    os.remove(STRIPPED)
shutil.copy2(SOURCE, STRIPPED)

con = sqlite3.connect(STRIPPED)
cur = con.cursor()

# Flush any WAL journal into the main file first
cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")

# ---- 2. Record before-stats ----------------------------------------------------
total_before = one(cur, "SELECT COUNT(*) FROM chunks")
with_emb = one(cur, "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
ents_before = one(cur, "SELECT COUNT(*) FROM chunk_entities")
print(f"Chunks: {total_before:,} | with embeddings stored: {with_emb:,} | "
      f"entity rows: {ents_before:,}")

if with_emb == 0:
    con.close()
    os.remove(STRIPPED)
    raise SystemExit(
        "No embeddings are stored in this database — nothing to strip.\n"
        "Your 16GB is text + FTS5 index + entity graph; keep the file as-is."
    )

# ---- 3. Strip embeddings --------------------------------------------------------
print("Stripping embedding column (safe: never read back by the SQLite path)...")
t0 = time.time()
cur.execute("UPDATE chunks SET embedding = NULL")
con.commit()
print(f"  done in {time.time() - t0:.0f}s")

# ---- 4. VACUUM -------------------------------------------------------------------
print("VACUUMing (10-30 minutes)...")
t0 = time.time()
cur.execute("VACUUM")
print(f"  VACUUM took {time.time() - t0:.0f}s")

# ---- 5. Verify nothing else changed ---------------------------------------------
total_after = one(cur, "SELECT COUNT(*) FROM chunks")
ents_after = one(cur, "SELECT COUNT(*) FROM chunk_entities")
docs_after = one(cur, "SELECT COUNT(*) FROM documents")
con.close()

assert total_after == total_before, "chunk count changed — do not use this file!"
assert ents_after == ents_before, "entity count changed — do not use this file!"

print(
    f"\nDone.\n"
    f"  Stripped DB: {STRIPPED} ({size_gb(STRIPPED):.2f} GB, "
    f"{size_gb(STRIPPED) / size_gb(SOURCE) * 100:.0f}% of original)\n"
    f"  Chunks: {total_after:,} (unchanged)\n"
    f"  Entity rows: {ents_after:,} (unchanged)\n"
    f"  Documents: {docs_after:,}"
)

print(
    "\nSanity check before relying on it — run the same query against both DBs and\n"
    "compare answers:\n"
    '  python -c "from v24 import build_large_corpus_engine; '
    "db,p,e = build_large_corpus_engine(); "
    "r = e.query_hybrid('What GitHub PRs relate to the Proxima Bank 429 incident?'); "
    'print(r[\'answer\'][:200])"\n'
    "\n(To test the stripped file, temporarily rename it to hybrid_rag_v22.db\n"
    "after moving the original aside.)"
)
