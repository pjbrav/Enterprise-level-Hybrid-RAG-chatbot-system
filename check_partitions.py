import sqlite3, sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "hybrid_rag.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("""
    SELECT partition_id, COUNT(*) as chunk_count
    FROM chunks
    GROUP BY partition_id
    ORDER BY chunk_count DESC
""")
rows = cur.fetchall()
total = sum(r[1] for r in rows)
print(f"{'Partition':<30} {'Chunks':>10} {'% of total':>10}")
print("-" * 52)
for partition_id, count in rows:
    print(f"{partition_id:<30} {count:>10,} {100*count/total:>9.1f}%")
print("-" * 52)
print(f"{'TOTAL':<30} {total:>10,}")
print(f"\nLargest partition: {rows[0][0]} with {rows[0][1]:,} chunks")