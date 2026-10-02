import pandas as pd
from pathlib import Path

df = pd.read_parquet("test.parquet")
base = Path("EnterpriseRAG_Docs")
base.mkdir(exist_ok=True)

for source, group in df.groupby("source_type"):
    folder = base / source.replace(" ", "_")
    folder.mkdir(exist_ok=True)
    for _, row in group.iterrows():
        doc_id = row.get("doc_id", f"doc_{_}")
        content = row.get("content", "")
        (folder / f"{doc_id}.txt").write_text(content, encoding="utf-8")

print("Done. Extracted to EnterpriseRAG_Docs/")