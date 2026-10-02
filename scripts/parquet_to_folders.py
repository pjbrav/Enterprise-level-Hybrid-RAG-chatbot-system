"""Rebuild the EnterpriseRAG_Docs folder tree from the raw dataset parquet.

Historical helper: this is how the corpus ingested into data/hybrid_rag_v22.db
was produced. The raw parquet (test.parquet, 1.4 GB) is NOT kept in the repo;
pass its path as the first argument if you ever need to rebuild:

    python scripts/parquet_to_folders.py path/to/test.parquet [output_dir]

Output defaults to ../datasets/EnterpriseRAG_Docs (outside the repo).
"""
import sys
from pathlib import Path

import pandas as pd

if len(sys.argv) < 2:
    sys.exit(__doc__)

df = pd.read_parquet(sys.argv[1])
base = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent.parent / "datasets" / "EnterpriseRAG_Docs"
base.mkdir(parents=True, exist_ok=True)

for source, group in df.groupby("source_type"):
    folder = base / source.replace(" ", "_")
    folder.mkdir(exist_ok=True)
    for _, row in group.iterrows():
        doc_id = row.get("doc_id", f"doc_{_}")
        content = row.get("content", "")
        (folder / f"{doc_id}.txt").write_text(content, encoding="utf-8")

print(f"Done. Extracted to {base}")
