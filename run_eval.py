"""
Run evaluation on the enterprise eval set (direct_retrieval.json, multi_hop.json, synthesis_contradiction.json).
Compares Standard RAG vs Hybrid RAG on each question.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime   # <-- added this

from v22 import DatabaseManager, DocumentProcessor, ComparisonEngine


def load_eval_questions(eval_dir: Path) -> List[Dict[str, Any]]:
    """Load all questions from the three JSON files."""
    questions = []
    for json_file in eval_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                item["source_file"] = json_file.name
                questions.append(item)
    return questions


def run_single_question(engine: ComparisonEngine, question: Dict[str, Any],
                        provider: str = "openai", api_key: str = None) -> Dict[str, Any]:
    """Run both Standard and Hybrid pipelines for one question."""
    q = question["question"]
    results = engine.compare(q, provider=provider, api_key=api_key)
    return {
        "eval_id": question["eval_id"],
        "complexity": question["complexity_level"],
        "domain": question["domain"],
        "standard": results["standard_rag"],
        "hybrid": results["hybrid_rag"],
    }


def evaluate_all(eval_dir: Path, output_file: Path, provider: str = "openai",
                 api_key: str = None, limit: int = None):
    db = DatabaseManager()
    processor = DocumentProcessor(db)
    engine = ComparisonEngine(processor, db, provider=provider, api_key=api_key)

    questions = load_eval_questions(eval_dir)
    if limit:
        questions = questions[:limit]

    print(f"Running {len(questions)} questions...")
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['eval_id']} ... ", end="", flush=True)
        try:
            res = run_single_question(engine, q, provider=provider, api_key=api_key)
            results.append(res)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            continue

    # Write a markdown report
    lines = ["# Enterprise RAG Evaluation Report", "",
             f"Run on {datetime.now().isoformat()}", "",
             "| Eval ID | Complexity | Domain | Std Latency | Std Tokens | Std Relevance | Hybrid Latency | Hybrid Tokens | Hybrid Relevance |",
             "|---------|------------|--------|-------------|------------|---------------|----------------|---------------|------------------|"]
    for r in results:
        s = r["standard"]
        h = r["hybrid"]
        lines.append(
            f"| {r['eval_id']} | {r['complexity']} | {r['domain'][:20]} | "
            f"{s['latency']:.2f}s | {s['tokens_used']} | {s['relevance_score']:.1f}% | "
            f"{h['latency']:.2f}s | {h['tokens_used']} | {h['relevance_score']:.1f}% |"
        )
    lines.append("\n## Summary")
    if results:
        std_lat = sum(r["standard"]["latency"] for r in results) / len(results)
        hyb_lat = sum(r["hybrid"]["latency"] for r in results) / len(results)
        std_rel = sum(r["standard"]["relevance_score"] for r in results) / len(results)
        hyb_rel = sum(r["hybrid"]["relevance_score"] for r in results) / len(results)
        lines.append(f"- **Average latency:** Standard {std_lat:.2f}s, Hybrid {hyb_lat:.2f}s")
        lines.append(f"- **Average relevance:** Standard {std_rel:.1f}%, Hybrid {hyb_rel:.1f}%")
        lines.append(f"- **Total questions:** {len(results)}")
    else:
        lines.append("No results were generated.")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, help="Directory containing the JSON eval files")
    parser.add_argument("--output", default="eval_report.md", help="Output markdown file")
    parser.add_argument("--provider", default="openai", choices=["openai", "claude", "gemini"])
    parser.add_argument("--api-key", help="API key (optional, if not set in .env)")
    parser.add_argument("--limit", type=int, help="Limit number of questions to run")
    args = parser.parse_args()

    evaluate_all(Path(args.eval_dir), Path(args.output), args.provider, args.api_key, args.limit)