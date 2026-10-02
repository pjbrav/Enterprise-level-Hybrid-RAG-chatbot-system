"""Interactive command-line entry point."""

from __future__ import annotations

import sys

from rag.db import DatabaseManager
from rag.ingest import DocumentProcessor, select_files_dialog
from rag.providers import choose_provider
from rag.engine import ComparisonEngine, export_markdown


def print_status(db: DatabaseManager, processor: DocumentProcessor) -> None:
    documents = db.get_all_documents()
    if processor.lazy:
        total_chunks = sum(db.get_partition_counts().values())
    else:
        total_chunks = len(processor.chunk_texts)
    print(f"\nLoaded documents: {len(documents)} | total chunks: {total_chunks}")
    for document in documents:
        print(
            f"  - {document['file_name']} | {document['partition_id']} | "
            f"{document['chunk_count']} chunks"
        )


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    print("=" * 72)
    print("FORCED HYBRID RAG v23 (AUTO‑PARTITIONING + DUMP‑AND‑GO)")
    print("TF-IDF + Entity Graph/RRF + Embeddings + Cross-Encoder (if available)")
    print("=" * 72)
    db = DatabaseManager()
    processor = DocumentProcessor(db)
    print_status(db, processor)
    provider = choose_provider()
    while True:
        print("\n1) Add papers with file picker")
        print("2) Add papers by path")
        print("3) Ask and compare")
        print("4) Show status")
        print("5) Quit")
        choice = input("Select (1-5): ").strip()
        if choice == "1":
            files = select_files_dialog()
            if files:
                processor.process_documents(files)
        elif choice == "2":
            raw_paths = input("Comma-separated PDF/TXT/MD/PY/JS/IPYNB/ZIP paths: ").strip()
            if raw_paths:
                processor.process_documents([path.strip() for path in raw_paths.split(",")])
        elif choice == "3":
            if not processor.chunk_texts:
                print("Load at least one document first.")
                continue
            query = input("\nQuestion: ").strip()
            if not query:
                continue
            engine = ComparisonEngine(processor, db, provider)
            results = engine.compare(query)
            print(engine.format_comparison(results))
            if input("Export Markdown report? (y/n): ").strip().lower() == "y":
                path = export_markdown(query, results)
                print(f"Exported: {path.resolve()}")
        elif choice == "4":
            print_status(db, processor)
        elif choice in {"5", "q", "quit", "exit"}:
            print("Goodbye.")
            return
        else:
            print("Please enter a number from 1 to 5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
