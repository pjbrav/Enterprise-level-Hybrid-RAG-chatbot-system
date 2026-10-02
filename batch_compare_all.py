"""
Batch comparison: Standard vs Hybrid RAG — ALL partitions, 200 questions.

Unlike batch_compare.py which filters to Jira+GitHub only, this runs
across ALL 500 questions in the parquet (up to SAMPLE_SIZE=200) regardless
of which partitions they touch. This gives a fair picture of whether
Hybrid outperforms Standard across the full corpus.

Usage:
    python batch_compare_all.py                 # 200 questions, all partitions
    python batch_compare_all.py --sample 50     # 50 questions (quick test)
    python batch_compare_all.py --sample 500    # all 500 questions
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from v23 import build_large_corpus_engine

# =======================================================
# CONFIG
# =======================================================
PARQUET_FILE = "questionaskerbigset.parquet"
SAMPLE_SIZE = 200
OUTPUT_CSV = "comparison_results_all.csv"
OUTPUT_MD = "comparison_report_all.md"
PROVIDER = "openai"

# Parse --sample arg
for i, arg in enumerate(sys.argv):
    if arg == "--sample" and i + 1 < len(sys.argv):
        try:
            SAMPLE_SIZE = int(sys.argv[i + 1])
        except ValueError:
            pass

# =======================================================
# PROGRESS BAR
# =======================================================
def print_progress(iteration, total, prefix="", suffix="", bar_length=50):
    if total == 0:
        return
    percent = 100 * iteration / float(total)
    filled_length = int(bar_length * iteration // total)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:.1f}% ({iteration}/{total}) {suffix}", end="", flush=True)
    if iteration >= total:
        print()

# =======================================================
# LOAD QUESTIONS — NO PARTITION FILTER
# =======================================================
print(f"\n{'=' * 70}")
print(f"  Standard vs Hybrid RAG — ALL PARTITIONS")
print(f"{'=' * 70}")

print("\n📖 Loading questions...")
df = pd.read_parquet(PARQUET_FILE)
print(f"Total questions in dataset: {len(df)}")

# Sample across ALL questions — no partition filter
if len(df) <= SAMPLE_SIZE:
    sampled_df = df
    print(f"Using all {len(df)} questions (sample size >= dataset)")
else:
    sampled_df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Sampled {len(sampled_df)} questions from {len(df)} (random_state=42)")

# Show what partitions are covered
if "source_types" in df.columns:
    all_sources = set()
    for val in df["source_types"].dropna():
        if isinstance(val, list):
            all_sources.update(str(s).lower().strip() for s in val if s)
        elif isinstance(val, str):
            all_sources.add(val.lower().strip())
    print(f"Source types in dataset: {sorted(all_sources)}")

# =======================================================
# INITIALISE ENGINE
# =======================================================
print("\n🔧 Initialising engine in LARGE CORPUS (lazy) mode...")
db, processor, engine = build_large_corpus_engine(
    db_path="hybrid_rag_v22.db",
    provider=PROVIDER,
)
print("✅ Engine ready in lazy mode (disk-backed FTS5 + entity tables).")

# =======================================================
# API KEY VALIDATION
# =======================================================
print("\n🔑 Checking API key...")
api_available = False
try:
    answer, info = engine._generate_answer(
        "Reply with exactly: OK",
        "",
        "API test",
        engine.provider_configs,
    )
    if info.get("provider"):
        print(f"✅ API key valid — using {info['provider'].title()}")
        api_available = True
    else:
        print("⚠️  API key failed — falling back to local extraction.")
except Exception as e:
    print(f"⚠️  API key validation error: {e}")
    print("   Falling back to local extraction.")

# =======================================================
# RUN COMPARISON
# =======================================================
print(f"\n🚀 Running comparison on {len(sampled_df)} questions...\n")
results = []
query_times = []
errors = 0

for idx, (_, row) in enumerate(sampled_df.iterrows(), 1):
    qid = row.get("question_id", f"q_{idx}")
    question = str(row["question"])

    query_start = time.time()
    print_progress(idx, len(sampled_df), prefix="Overall")
    print(f"  [{idx}/{len(sampled_df)}] {qid} ... ", end="", flush=True)

    try:
        res = engine.compare(question, provider=PROVIDER, include_direct_ai=False)
        query_time = time.time() - query_start
        query_times.append(query_time)

        std = res["standard_rag"]
        hyb = res["hybrid_rag"]
        router = hyb.get("router_decision", {})

        gen_info = hyb.get("generation", {})
        if gen_info.get("provider"):
            status = f"✓ ({query_time:.1f}s)"
        else:
            status = f"⚠️ local ({query_time:.1f}s)"

        # Determine which partitions were retrieved from
        hyb_sources = hyb.get("sources", [])
        std_sources = std.get("sources", [])

        results.append({
            "question_id": qid,
            "question": question[:120],
            "question_type": row.get("question_type", ""),
            "std_latency": round(std["latency"], 2),
            "std_tokens": std["tokens_used"],
            "std_relevance": round(std["relevance_score"], 1),
            "std_chunks": std["chunks_retrieved"],
            "std_confidence": round(std["confidence_score"], 3),
            "std_answer_relevance": round(std.get("answer_relevance", 0), 1),
            "hyb_latency": round(hyb["latency"], 2),
            "hyb_tokens": hyb["tokens_used"],
            "hyb_relevance": round(hyb["relevance_score"], 1),
            "hyb_chunks": hyb["chunks_retrieved"],
            "hyb_confidence": round(hyb["confidence_score"], 3),
            "hyb_cross_domain": hyb.get("cross_domain", False),
            "hyb_quality_validated": hyb.get("quality_validated", False),
            "hyb_answer_relevance": round(hyb.get("answer_relevance", 0), 1),
            "api_used": bool(gen_info.get("provider")),
            "router_chosen": router.get("chosen", "unknown"),
            "router_reason": router.get("reason", "")[:80],
        })
        print(status)
    except Exception as e:
        query_time = time.time() - query_start
        errors += 1
        print(f"✗ ({query_time:.1f}s) {str(e)[:80]}")
        continue

# =======================================================
# SUMMARY
# =======================================================
print(f"\n{'=' * 70}")
print(f"  RESULTS SUMMARY")
print(f"{'=' * 70}")

if not results:
    print("No results to save.")
    sys.exit(1)

avg_std_rel = sum(r["std_relevance"] for r in results) / len(results)
avg_hyb_rel = sum(r["hyb_relevance"] for r in results) / len(results)
avg_std_lat = sum(r["std_latency"] for r in results) / len(results)
avg_hyb_lat = sum(r["hyb_latency"] for r in results) / len(results)
total_std_tokens = sum(r["std_tokens"] for r in results)
total_hyb_tokens = sum(r["hyb_tokens"] for r in results)
token_diff = total_hyb_tokens - total_std_tokens
token_diff_pct = (token_diff / max(1, total_std_tokens)) * 100

# Win/loss/tie
hybrid_wins = sum(1 for r in results if r["hyb_relevance"] > r["std_relevance"] + 1)
standard_wins = sum(1 for r in results if r["std_relevance"] > r["hyb_relevance"] + 1)
ties = len(results) - hybrid_wins - standard_wins

cross_domain_count = sum(1 for r in results if r["hyb_cross_domain"])
quality_validated_count = sum(1 for r in results if r["hyb_quality_validated"])

print(f"\n  Questions run:          {len(results)}")
print(f"  Errors:                {errors}")
print(f"  Avg query time:        {sum(query_times)/len(query_times):.1f}s")
print(f"  Fastest / Slowest:     {min(query_times):.1f}s / {max(query_times):.1f}s")
print()
print(f"  Relevance (chunk-level):")
print(f"    Standard avg:        {avg_std_rel:.1f}%")
print(f"    Hybrid avg:          {avg_hyb_rel:.1f}%")
print(f"    Difference:          {avg_hyb_rel - avg_std_rel:+.1f} pp")
print()
print(f"  Win/Loss/Tie (±1% threshold):")
print(f"    Hybrid wins:         {hybrid_wins} ({hybrid_wins/len(results)*100:.0f}%)")
print(f"    Standard wins:       {standard_wins} ({standard_wins/len(results)*100:.0f}%)")
print(f"    Ties:                {ties} ({ties/len(results)*100:.0f}%)")
print()
print(f"  Latency:")
print(f"    Standard avg:        {avg_std_lat:.1f}s")
print(f"    Hybrid avg:          {avg_hyb_lat:.1f}s")
print(f"    Hybrid overhead:     {avg_hyb_lat - avg_std_lat:+.1f}s")
print()
print(f"  Tokens:")
print(f"    Standard total:      {total_std_tokens:,}")
print(f"    Hybrid total:        {total_hyb_tokens:,}")
print(f"    Difference:          {token_diff:+,} ({token_diff_pct:+.1f}%)")
print()
print(f"  Cross-domain triggered: {cross_domain_count}/{len(results)} ({cross_domain_count/len(results)*100:.0f}%)")
print(f"  Quality validated:       {quality_validated_count}/{len(results)} ({quality_validated_count/len(results)*100:.0f}%)")

# Save CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\n📊 CSV saved to {OUTPUT_CSV}")

# Save Markdown report
lines = [
    "# Standard vs Hybrid RAG — All Partitions",
    f"Run on {datetime.now().isoformat()}",
    f"Total questions: {len(results)}",
    f"Engine mode: Lazy (disk-backed)",
    f"API available: {'✅' if api_available else '❌'}",
    "",
    "## Summary",
    f"| Metric | Standard | Hybrid | Difference |",
    f"|--------|----------|--------|------------|",
    f"| Avg relevance | {avg_std_rel:.1f}% | {avg_hyb_rel:.1f}% | {avg_hyb_rel - avg_std_rel:+.1f} pp |",
    f"| Avg latency | {avg_std_lat:.1f}s | {avg_hyb_lat:.1f}s | {avg_hyb_lat - avg_std_lat:+.1f}s |",
    f"| Total tokens | {total_std_tokens:,} | {total_hyb_tokens:,} | {token_diff:+,} |",
    "",
    f"- **Hybrid wins:** {hybrid_wins}/{len(results)} ({hybrid_wins/len(results)*100:.0f}%)",
    f"- **Standard wins:** {standard_wins}/{len(results)} ({standard_wins/len(results)*100:.0f}%)",
    f"- **Ties:** {ties}/{len(results)} ({ties/len(results)*100:.0f}%)",
    f"- **Cross-domain triggered:** {cross_domain_count}/{len(results)}",
    "",
    "## Per-question results",
    "| QID | Type | Std Rel | Hyb Rel | Diff | Std Tok | Hyb Tok | Cross-Domain |",
    "|-----|------|---------|---------|------|---------|---------|--------------|",
]
for r in results:
    diff = r["hyb_relevance"] - r["std_relevance"]
    cd = "✅" if r["hyb_cross_domain"] else ""
    lines.append(
        f"| {r['question_id']} | {r['question_type']} | "
        f"{r['std_relevance']:.1f}% | {r['hyb_relevance']:.1f}% | "
        f"{diff:+.1f} | {r['std_tokens']} | {r['hyb_tokens']} | {cd} |"
    )

Path(OUTPUT_MD).write_text("\n".join(lines), encoding="utf-8")
print(f"📄 Markdown report saved to {OUTPUT_MD}")
print(f"\n{'=' * 70}")
