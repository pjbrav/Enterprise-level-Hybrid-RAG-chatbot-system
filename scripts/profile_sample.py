"""
Profiles 5 randomly-sampled questions (fresh sample every run) from your
real questions parquet, breaking each one down by stage: partition
routing, fts_search, entity_search (with per-entity hit counts, so a
high-frequency entity shows up clearly), and the generation API call --
including which provider actually answered it, so provider fallback
chains are visible rather than hidden inside a single latency number.

Usage:
    python scripts/profile_sample.py data/hybrid_rag_v22.db data/questionaskerbigset.parquet
    python scripts/profile_sample.py data/hybrid_rag_v22.db data/questionaskerbigset.parquet --n 5 --provider openai
    python scripts/profile_sample.py data/hybrid_rag_v22.db data/questionaskerbigset.parquet --seed 7   # reproducible sample instead of random
"""
import argparse
import random
import sys
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import v24 as v23


def timed(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"    {label}: {elapsed:.2f}s")
    return result, elapsed


def profile_one(engine, db, query: str) -> dict:
    print(f"\n{'-' * 78}")
    print(f"Q: {query}")
    print("-" * 78)

    partitions, t_partitions = timed("_get_active_partitions", engine._get_active_partitions, query)
    print(f"    -> active partitions: {partitions}")

    terms = v23.query_terms(query)
    entities = v23.extract_entities(query)
    print(f"    -> query entities: {entities}")

    # Reference-only: per-entity hit counts, for diagnosing WHICH entity is
    # expensive. These are extra, unbounded-limit probes that do NOT run in
    # production -- reported separately, never summed into the timing
    # numbers below, so they can't distort the retrieval/generation split.
    entity_hit_counts = {}
    scopes = partitions if partitions else [None]
    for partition_id in scopes:
        label = partition_id or "(global)"
        if entities:
            for entity in entities:
                rows, _ = timed(f"    [reference only] '{entity}' hit count in {label}",
                                 db.entity_search, {entity}, partition_id, 1_000_000)
                entity_hit_counts[(entity, label)] = len(rows)

    # Real retrieval cost, broken down per-partition, with explicit
    # visibility into the WEAK_TOP_SCORE global-fallback branch --
    # confirmed via profiling to potentially run an UNFILTERED (whole
    # corpus) search per weak partition, which can dwarf the cost of the
    # partition-scoped searches it's supposed to be a rare fallback for.
    t_cross_domain_start = time.perf_counter()
    active_partitions_for_ranking = engine._get_active_partitions(query)
    partition_breakdown = []
    if not active_partitions_for_ranking:
        _, t_global = timed("  no active partitions -> single global _rank_candidates call",
                             engine._rank_candidates, query, None, True)
    else:
        for partition_id in active_partitions_for_ranking:
            print(f"  [{partition_id}] -- breaking _rank_candidates into its actual sub-steps:")
            fts_raw, t_fts = timed(f"    fts_search ({len(terms)} terms: {terms})",
                                    db.fts_search, terms, partition_id, engine.LEXICAL_CANDIDATE_LIMIT)
            lexical, t_calib = timed("    _calibrate_lexical_scores", engine._calibrate_lexical_scores, fts_raw)
            graph = []
            t_entity = 0.0
            if entities:
                graph, t_entity = timed(f"    entity_search (combined, {len(entities)} entities)",
                                         db.entity_search, entities, partition_id, engine.LEXICAL_CANDIDATE_LIMIT)
            if graph:
                ranking, t_fuse = timed("    _fuse_rankings", engine._fuse_rankings, lexical, graph)
            else:
                ranking, t_fuse = lexical, 0.0
            t_scoped = t_fts + t_calib + t_entity + t_fuse
            print(f"    -> partition total: {t_scoped:.2f}s  (fts={t_fts:.2f}s, entity={t_entity:.2f}s, fuse={t_fuse:.2f}s)")
            top_score = ranking[0][1] if ranking else None
            triggers_fallback = (not ranking) or (top_score is not None and top_score <= engine.WEAK_TOP_SCORE)
            print(f"    -> top score: {top_score}  (WEAK_TOP_SCORE={engine.WEAK_TOP_SCORE})  "
                  f"triggers global fallback: {triggers_fallback}")
            t_fallback = 0.0
            if triggers_fallback:
                _, t_fallback = timed(f"    GLOBAL FALLBACK for [{partition_id}] (unfiltered, whole corpus)",
                                       engine._rank_candidates, query, None, True)
            partition_breakdown.append((partition_id, t_scoped, triggers_fallback, t_fallback))
    t_retrieval = time.perf_counter() - t_cross_domain_start
    print(f"    TOTAL cross-domain retrieval (reconstructed): {t_retrieval:.2f}s")
    fallback_total = sum(fb for _, _, _, fb in partition_breakdown)
    if fallback_total > 0:
        print(f"    -> of which {fallback_total:.2f}s ({100*fallback_total/max(t_retrieval,0.001):.0f}%) "
              f"was spent on global-fallback (unfiltered) searches")

    t0 = time.perf_counter()
    result = engine._hybrid_rag(query, engine.provider_configs)
    t_full = time.perf_counter() - t0
    provider_used = result["generation"].get("provider")
    notice = result["generation"].get("notice", "")
    print(f"    Full _hybrid_rag call: {t_full:.2f}s  (provider used: {provider_used or 'NONE -- local fallback'})")
    if notice:
        print(f"    notice: {notice[:200]}")

    # generation_time is derived from t_full (the real _hybrid_rag call)
    # minus t_retrieval (the real _get_cross_domain_rankings call, timed
    # separately but calling the identical method with identical
    # arguments) -- both are production-path costs, so this split is
    # trustworthy, unlike subtracting unrelated diagnostic probe time.
    # It's still an approximation (retrieval runs twice: once here, once
    # again inside _hybrid_rag -- SQLite/OS caching may make the second
    # call slightly faster, so generation_time may be a slight
    # overestimate), but no longer contaminated by the unbounded
    # reference-only entity probes above.
    retrieval_time = t_retrieval
    generation_time = max(0.0, t_full - retrieval_time)

    # Flag the biggest single contributor for a quick read at a glance
    worst_entity = max(entity_hit_counts.items(), key=lambda kv: kv[1], default=(None, 0))

    return {
        "query": query,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": t_full,
        "provider_used": provider_used,
        "worst_entity": worst_entity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("questions_path")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--question-col", default="question")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--seed", type=int, default=None, help="Omit for a fresh random sample every run")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.questions_path)
    if args.question_col not in df.columns:
        print(f"Column '{args.question_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)
    print(f"Sampling {args.n} questions (seed={seed})")
    sample = df.sample(n=min(args.n, len(df)), random_state=seed)

    print(f"Building engine (provider={args.provider})...")
    db, processor, engine = v23.build_large_corpus_engine(db_path=args.db_path, provider=args.provider)
    print(f"provider_configs resolved to: {[c.name for c in engine.provider_configs]}")
    if len(engine.provider_configs) > 1:
        print(f"  NOTE: more than one provider is configured -- if '{args.provider}' fails, "
              f"it WILL fall through to: {[c.name for c in engine.provider_configs[1:]]}, "
              f"adding latency per attempt.")

    rows = [profile_one(engine, db, str(q)) for q in sample[args.question_col]]

    print(f"\n{'=' * 78}")
    print("SUMMARY ACROSS THIS SAMPLE")
    print("=" * 78)
    for r in rows:
        flag = ""
        if r["generation_time"] > r["retrieval_time"] * 3:
            flag = " <- generation-dominated"
        elif r["retrieval_time"] > 5:
            flag = f" <- retrieval-dominated (worst entity: {r['worst_entity']})"
        print(f"  total={r['total_time']:6.2f}s  retrieval={r['retrieval_time']:6.2f}s  "
              f"generation={r['generation_time']:6.2f}s  provider={r['provider_used'] or 'LOCAL'}{flag}")

    avg_total = sum(r["total_time"] for r in rows) / len(rows)
    avg_retrieval = sum(r["retrieval_time"] for r in rows) / len(rows)
    avg_generation = sum(r["generation_time"] for r in rows) / len(rows)
    local_count = sum(1 for r in rows if not r["provider_used"])
    print(f"\n  Avg total: {avg_total:.2f}s | Avg retrieval: {avg_retrieval:.2f}s | Avg generation: {avg_generation:.2f}s")
    print(f"  Fell back to local (no provider answered): {local_count}/{len(rows)}")
    if local_count > 0:
        print("  -> If this is nonzero and you expect real API answers, check provider_configs "
              "above and your OPENAI_MODEL env var -- see notices printed per-question.")


if __name__ == "__main__":
    main()
