"""
Build a smaller demo database for web hosting (Hugging Face Spaces).

What it does:
  1. Copies hybrid_rag_v22.db -> hybrid_rag_demo.db (original is untouched)
  2. Strips the embedding column (biggest space saver; the SQLite path never
     reads it back — FTS5 + entity graph + cross-encoder don't need it)
  3. Keeps only the partitions listed in KEEP below
  4. Rebuilds entity_frequency to match
  5. VACUUMs to reclaim the space

Requirements: ~16GB free disk (for the copy) and 10-30 minutes.
Usage:  python make_demo_db.py
"""

import os
import shutil
import sqlite3
import time

SOURCE = "data/hybrid_rag_v22.db"
DEMO = "data/hybrid_rag_demo.db"

# Partitions to keep in the demo. Edit after seeing the printed list below.
KEEP = ["partition_github", "partition_jira", "partition_slack"]


def size_gb(path: str) -> float:
    return os.path.getsize(path) / 1e9


def one(cur, sql, args=()):
    return cur.execute(sql, args).fetchone()[0]


# ---- 0. Inspect the source database ---------------------------------------
print(f"Source: {SOURCE} ({size_gb(SOURCE):.2f} GB)")

con = sqlite3.connect(SOURCE)
cur = con.cursor()
total = one(cur, "SELECT COUNT(*) FROM chunks")
with_emb = one(cur, "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
print(f"Chunks: {total:,} | with embeddings stored: {with_emb:,}")

print("\nPartitions in source DB:")
partitions = cur.execute(
    "SELECT partition_id, COUNT(*) FROM chunks GROUP BY partition_id ORDER BY COUNT(*) DESC"
).fetchall()
for pid, cnt in partitions:
    keep_flag = "KEEP" if pid in KEEP else "drop"
    print(f"  {pid:30s} {cnt:>9,}  [{keep_flag}]")
con.close()

missing = [p for p in KEEP if p not in {pid for pid, _ in partitions}]
if missing:
    raise SystemExit(f"ERROR: these KEEP partitions don't exist: {missing}. Edit KEEP and re-run.")

# ---- 1. Copy ---------------------------------------------------------------
print(f"\nCopying {SOURCE} -> {DEMO} ...")
if os.path.exists(DEMO):
    os.remove(DEMO)
shutil.copy2(SOURCE, DEMO)

con = sqlite3.connect(DEMO)
cur = con.cursor()
ph = ",".join("?" * len(KEEP))

# ---- 2. Strip embeddings ---------------------------------------------------
print("Stripping embedding column (safe: SQLite path never reads it back)...")
t0 = time.time()
cur.execute("UPDATE chunks SET embedding = NULL")
con.commit()
print(f"  done in {time.time() - t0:.0f}s")

# ---- 3. Keep only chosen partitions ----------------------------------------
print(f"Deleting chunks outside: {KEEP}")
cur.execute(f"DELETE FROM chunks WHERE partition_id NOT IN ({ph})", KEEP)
print(f"  chunks left: {one(cur, 'SELECT COUNT(*) FROM chunks'):,}")

print("Cleaning FTS5 index...")
cur.execute(f"DELETE FROM chunks_fts WHERE partition_id NOT IN ({ph})", KEEP)

print("Cleaning entity graph...")
cur.execute(f"DELETE FROM chunk_entities WHERE partition_id NOT IN ({ph})", KEEP)
print(f"  entity rows left: {one(cur, 'SELECT COUNT(*) FROM chunk_entities'):,}")

print("Cleaning documents...")
cur.execute(f"DELETE FROM documents WHERE partition_id NOT IN ({ph})", KEEP)
con.commit()

# ---- 4. Rebuild entity_frequency --------------------------------------------
print("Rebuilding entity_frequency...")
cur.execute("DELETE FROM entity_frequency")
cur.execute(
    "INSERT INTO entity_frequency (entity, freq) "
    "SELECT entity, COUNT(*) FROM chunk_entities GROUP BY entity"
)
con.commit()

# ---- 5. VACUUM ---------------------------------------------------------------
print("VACUUMing (the slow part — 5-20 minutes)...")
t0 = time.time()
cur.execute("VACUUM")
print(f"  VACUUM took {time.time() - t0:.0f}s")

kept = one(cur, "SELECT COUNT(*) FROM chunks")
ents = one(cur, "SELECT COUNT(*) FROM chunk_entities")
con.close()

print(
    f"\nDone.\n"
    f"  Demo DB : {DEMO} ({size_gb(DEMO):.2f} GB, "
    f"{size_gb(DEMO) / size_gb(SOURCE) * 100:.0f}% of original)\n"
    f"  Chunks  : {kept:,} / {total:,}\n"
    f"  Entities: {ents:,}"
)
print(
    "\nNext: upload hybrid_rag_demo.db to your HF Space AS 'hybrid_rag_v22.db'\n"
    "(same filename the app expects — replaces the full DB there only)."
)
