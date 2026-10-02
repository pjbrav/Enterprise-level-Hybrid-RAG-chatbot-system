import sqlite3
import os

db_path = 'hybrid_rag_v22.db'

if not os.path.exists(db_path):
    print("❌ Database not found.")
    exit()

size_mb = os.path.getsize(db_path) / (1024 * 1024)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("PRAGMA integrity_check")
integrity = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM documents")
docs = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM chunks")
chunks = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(DISTINCT file_path) FROM documents")
unique_paths = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(DISTINCT content_hash) FROM documents")
unique_hashes = cursor.fetchone()[0]

cursor.execute("SELECT file_path, COUNT(*) FROM documents GROUP BY file_path HAVING COUNT(*) > 1 LIMIT 1")
dup_paths = cursor.fetchone() is not None

cursor.execute("SELECT content_hash, COUNT(*) FROM documents GROUP BY content_hash HAVING COUNT(*) > 1 LIMIT 1")
dup_hashes = cursor.fetchone() is not None

conn.close()

print("📊 Database Health Check:")
print(f"   Integrity: {integrity}")
print(f"   Documents: {docs:,}")
print(f"   Chunks: {chunks:,}")
print(f"   Size: {size_mb:.2f} MB")
print(f"   Unique file paths: {unique_paths:,}")
print(f"   Unique content hashes: {unique_hashes:,}")
print("   Duplicate paths: " + ("⚠️ YES" if dup_paths else "✅ No"))
print("   Duplicate hashes: " + ("⚠️ YES" if dup_hashes else "✅ No"))