"""
One-time backfill for entity_frequency -- required for entity_search's
new high-fanout filtering to actually take effect on your existing 34M-row
chunk_entities table (it silently no-ops until this table is populated).

Usage:
    python backfill_entity_frequency.py your_database.db

Safe to re-run any time (e.g. after re-running backfill_index.py for a
new partition) -- it replaces the whole table each time from current
chunk_entities contents.
"""
import sqlite3
import sys
import time


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    db_path = sys.argv[1]

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""CREATE TABLE IF NOT EXISTS entity_frequency (entity TEXT PRIMARY KEY, freq INTEGER NOT NULL)""")

    print("Computing entity frequencies (one GROUP BY over chunk_entities -- "
          "this is the expensive one-time cost; queries after this are cheap)...")
    t0 = time.perf_counter()
    conn.execute("DELETE FROM entity_frequency")
    conn.execute(
        "INSERT INTO entity_frequency (entity, freq) "
        "SELECT entity, COUNT(*) FROM chunk_entities GROUP BY entity"
    )
    conn.commit()
    elapsed = time.perf_counter() - t0

    row_count = conn.execute("SELECT COUNT(*) FROM entity_frequency").fetchone()[0]
    print(f"Done in {elapsed:.1f}s. {row_count:,} unique entities recorded.")

    print("\nTop 20 highest-frequency entities (these will be excluded from "
          "entity_search once above MAX_ENTITY_FANOUT, default 5000):")
    for entity, freq in conn.execute("SELECT entity, freq FROM entity_frequency ORDER BY freq DESC LIMIT 20"):
        flag = "  <- will be filtered (over 5000)" if freq > 5000 else ""
        print(f"  {freq:>10,}  {entity}{flag}")

    conn.close()


if __name__ == "__main__":
    main()
