"""
Run evaluation using v23 (forced hybrid) on the enterprise eval set.
Outputs a report with latency, tokens, and relevance for hybrid RAG.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Use v23 instead of v22
from v23 import DatabaseManager, DocumentProcessor, ComparisonEngine


def load_questions(eval_dir: Path) -> List[Dict[str, Any]]:
    questions = []
    for json_file in eval_dir.rglob("*.json"):
        # Skip known non‑question files
        if json_file.name.lower() == "schema.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # If it's a list, process each item
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item["source_file"] = json_file.name
                        questions.append(item)
                    else:
                        print(f"Warning: {json_file.name} contains non‑dict item, skipping.")
            else:
                print(f"Warning: {json_file.name} is not a list, skipping.")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    return questions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, help="Directory containing the eval JSONs (with subfolders)")
    parser.add_argument("--provider", default="openai", choices=["openai", "claude", "gemini"])
    parser.add_argument("--api-key", help="API key (optional, if not in .env)")
    parser.add_argument("--output", default="eval_hybrid_report.md", help="Output markdown file")
    args = parser.parse_args()

    db = DatabaseManager()
    processor = DocumentProcessor(db)
    engine = ComparisonEngine(processor, db, provider=args.provider, api_key=args.api_key)

    questions = load_questions(Path(args.eval_dir))
    print(f"Loaded {len(questions)} questions.")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q.get('eval_id', 'unknown')} ... ", end="", flush=True)
        try:
            res = engine.compare_single(q["question"], method="hybrid_rag",
                                        provider=args.provider, api_key=args.api_key)
            results.append({
                "eval_id": q.get("eval_id", "unknown"),
                "complexity": q.get("complexity_level", 0),
                "domain": q.get("domain", "")[:20],
                "latency": res["latency"],
                "tokens": res["tokens_used"],
                "relevance": res["relevance_score"],
                "chunks": res["chunks_retrieved"],
                "confidence": res["confidence_score"]
            })
            print("✓")
        except Exception as e:
            print(f"✗ {e}")

    # Write report
    lines = ["# Forced Hybrid RAG Evaluation (v23)", "",
             f"Run on {datetime.now().isoformat()}", "",
             "| Eval ID | Complexity | Domain | Latency (s) | Tokens | Relevance (%) | Chunks | Confidence |",
             "|---------|------------|--------|-------------|--------|---------------|--------|------------|"]
    for r in results:
        lines.append(
            f"| {r['eval_id']} | {r['complexity']} | {r['domain']} | "
            f"{r['latency']:.2f} | {r['tokens']} | {r['relevance']:.1f} | "
            f"{r['chunks']} | {r['confidence']:.2f} |"
        )
    lines.append("\n## Summary")
    if results:
        avg_lat = sum(r["latency"] for r in results) / len(results)
        avg_tok = sum(r["tokens"] for r in results) / len(results)
        avg_rel = sum(r["relevance"] for r in results) / len(results)
        lines.append(f"- **Average latency:** {avg_lat:.2f}s")
        lines.append(f"- **Average tokens:** {avg_tok:.0f}")
        lines.append(f"- **Average relevance:** {avg_rel:.1f}%")
        lines.append(f"- **Total questions:** {len(results)}")
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {args.output}")

if __name__ == "__main__":
    main()