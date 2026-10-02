"""
Batch comparison: Standard vs Hybrid RAG
Filters questions involving BOTH Jira and GitHub partitions for a fair test.
"""

import pandas as pd
import time
import sys
import re
from pathlib import Path
from datetime import datetime
from v23 import DatabaseManager, DocumentProcessor, ComparisonEngine
from v23 import build_tfidf_index, EntityKnowledgeGraph, stem_token

# =======================================================
# CONFIGURATION
# =======================================================
PARQUET_FILE = 'questionaskerbigset.parquet'
SAMPLE_SIZE = 5
OUTPUT_CSV = 'comparison_results_jira_github.csv'
OUTPUT_MD = 'comparison_report_jira_github.md'
PROVIDER = 'openai'

# =======================================================
# PROGRESS BAR FUNCTION
# =======================================================
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    if total == 0:
        return
    percent = 100 * (iteration / float(total))
    filled_length = int(bar_length * iteration // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.{decimals}f}% {suffix}', end='')
    if iteration >= total:
        print()

# =======================================================
# LOAD AND FILTER QUESTIONS (auto-detects column format)
# =======================================================
print('\n📖 Loading questions...')
df = pd.read_parquet(PARQUET_FILE)
print(f'Total questions in dataset: {len(df)}')

def has_both_partitions(row):
    """Detect if a question involves both Jira and GitHub partitions."""
    # Method 1: Check source_types column
    sources = row.get('source_types')
    if sources is not None:
        if isinstance(sources, list):
            sources_set = {str(s).lower().strip() for s in sources if s}
        elif isinstance(sources, str):
            if sources.startswith('['):
                try:
                    items = sources.strip('[]').replace("'", "").replace('"', '').split(',')
                    sources_set = {item.strip().lower() for item in items if item.strip()}
                except:
                    sources_set = {sources.lower()}
            else:
                sources_set = {s.strip().lower() for s in sources.split(',') if s.strip()}
        else:
            sources_set = {str(sources).lower()}
        if 'jira' in sources_set and 'github' in sources_set:
            return True
    
    # Method 2: Check source_documents column
    docs = row.get('source_documents')
    if docs is not None:
        docs_str = str(docs).lower()
        if 'jira' in docs_str and 'github' in docs_str:
            return True
    
    # Method 3: Check expected_doc_ids
    doc_ids = row.get('expected_doc_ids')
    if doc_ids is not None:
        doc_str = str(doc_ids).lower()
        if ('jira' in doc_str or 'ticket' in doc_str) and ('github' in doc_str or 'pr' in doc_str):
            return True
    
    # Method 4: Check the question text itself
    question = row.get('question', '')
    if question:
        q = question.lower()
        has_jira = any(word in q for word in ['jira', 'ticket', 'issue', 'epic', 'sprint'])
        has_github = any(word in q for word in ['github', 'pr', 'pull request', 'repo', 'commit'])
        if has_jira and has_github:
            return True
    
    return False

# Apply filter
filtered_df = df[df.apply(has_both_partitions, axis=1)]
print(f'Questions involving BOTH jira and github: {len(filtered_df)}')

if len(filtered_df) == 0:
    # Fallback: use questions that mention either jira OR github
    print('⚠️ No questions explicitly marked as both Jira and GitHub.')
    print('   Falling back to questions that mention JIRA OR GITHUB...')
    
    def has_either_partition(row):
        sources = row.get('source_types')
        if sources is not None:
            if isinstance(sources, list):
                sources_set = {str(s).lower().strip() for s in sources if s}
            elif isinstance(sources, str):
                if sources.startswith('['):
                    try:
                        items = sources.strip('[]').replace("'", "").replace('"', '').split(',')
                        sources_set = {item.strip().lower() for item in items if item.strip()}
                    except:
                        sources_set = {sources.lower()}
                else:
                    sources_set = {s.strip().lower() for s in sources.split(',') if s.strip()}
            else:
                sources_set = {str(sources).lower()}
            if 'jira' in sources_set or 'github' in sources_set:
                return True
        doc_ids = row.get('expected_doc_ids')
        if doc_ids is not None:
            doc_str = str(doc_ids).lower()
            if 'jira' in doc_str or 'github' in doc_str:
                return True
        question = row.get('question', '')
        if question:
            q = question.lower()
            if 'jira' in q or 'github' in q:
                return True
        return False
    
    filtered_df = df[df.apply(has_either_partition, axis=1)]
    print(f'Questions involving JIRA OR GITHUB: {len(filtered_df)}')

if len(filtered_df) == 0:
    print('⚠️ No Jira/GitHub questions found. Using all questions as fallback.')
    sampled_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
elif len(filtered_df) < SAMPLE_SIZE:
    print(f'⚠️ Only {len(filtered_df)} questions available. Using all of them.')
    sampled_df = filtered_df
else:
    sampled_df = filtered_df.sample(n=SAMPLE_SIZE, random_state=42)

print(f'✅ Running on {len(sampled_df)} questions.')

# =======================================================
# INITIALISE ENGINE (Lazy mode for large corpus)
# =======================================================
print('\n🔧 Initialising engine in LARGE CORPUS (lazy) mode...')
from v23 import build_large_corpus_engine

db, processor, engine = build_large_corpus_engine(
    db_path="hybrid_rag_v22.db",
    provider=PROVIDER
)
print('✅ Engine ready in lazy mode (disk-backed FTS5 + entity tables).')

# =======================================================
# API KEY VALIDATION
# =======================================================
print('\n🔑 Checking API key...')
api_available = False
try:
    test_result = engine._generate_answer(
        "Reply with exactly: OK", 
        "", 
        "API test", 
        engine.provider_configs
    )
    answer, info = test_result
    if info.get('provider'):
        print(f'✅ API key valid – using {info["provider"].title()}')
        api_available = True
    else:
        print('⚠️  API key failed – falling back to local extraction.')
except Exception as e:
    print(f'⚠️  API key validation error: {e}')
    print('   Falling back to local extraction.')

# =======================================================
# RUN COMPARISON WITH PER‑QUERY TIMING
# =======================================================
print(f'\n🚀 Running comparison on {len(sampled_df)} questions...\n')
results = []
query_times = []

for idx, (_, row) in enumerate(sampled_df.iterrows(), 1):
    qid = row.get('question_id', f'q_{idx}')
    question = row['question']
    
    query_start = time.time()
    print(f'[{idx}/{len(sampled_df)}] {qid} ... ', end='', flush=True)
    
    try:
        res = engine.compare(question, provider=PROVIDER)
        query_time = time.time() - query_start
        query_times.append(query_time)
        
        std = res['standard_rag']
        hyb = res['hybrid_rag']
        router = hyb.get('router_decision', {})
        
        gen_info = hyb.get('generation', {})
        if gen_info.get('provider'):
            status = f'✓ ({query_time:.1f}s)'
        else:
            status = f'⚠️ local ({query_time:.1f}s)'
        
        results.append({
            'question_id': qid,
            'question_type': row.get('question_type', ''),
            'std_latency': std['latency'],
            'std_tokens': std['tokens_used'],
            'std_relevance': std['relevance_score'],
            'hyb_latency': hyb['latency'],
            'hyb_tokens': hyb['tokens_used'],
            'hyb_relevance': hyb['relevance_score'],
            'api_used': gen_info.get('provider', False),
            'router_chosen': router.get('chosen', 'unknown'),
            'router_confidence': router.get('standard_confidence', 0),
            'router_reason': router.get('reason', '')[:100],
        })
        print(status)
    except Exception as e:
        query_time = time.time() - query_start
        print(f'✗ ({query_time:.1f}s) {str(e)[:80]}')
        continue

# =======================================================
# SUMMARY STATISTICS
# =======================================================
if results:
    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    print(f'\n📊 Average query time: {avg_query_time:.1f}s')
    print(f'   Fastest: {min(query_times):.1f}s | Slowest: {max(query_times):.1f}s')
    
    # Router statistics
    hybrid_used = sum(1 for r in results if r['router_chosen'] == 'hybrid')
    standard_used = len(results) - hybrid_used
    print(f'   Router chose Hybrid: {hybrid_used}/{len(results)} ({hybrid_used/len(results)*100:.1f}%)')
    print(f'   Router chose Standard: {standard_used}/{len(results)} ({standard_used/len(results)*100:.1f}%)')
    
    # Calculate token savings
    total_std_tokens = sum(r['std_tokens'] for r in results)
    total_hyb_tokens = sum(r['hyb_tokens'] for r in results)
    token_savings = total_std_tokens - total_hyb_tokens
    token_savings_pct = (token_savings / total_std_tokens * 100) if total_std_tokens else 0
    
    print(f'   Total Standard tokens: {total_std_tokens:,}')
    print(f'   Total Hybrid tokens: {total_hyb_tokens:,}')
    print(f'   Token savings: {token_savings:,} ({token_savings_pct:.1f}%)')
    
    import csv
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'📊 Results saved to {OUTPUT_CSV}')

    # Generate Markdown report
    lines = [
        '# Standard vs Hybrid RAG Comparison',
        f'## Dataset: Jira + GitHub focused',
        f'Run on {datetime.now().isoformat()}',
        f'Total questions: {len(results)}',
        f'Engine mode: Lazy (disk-backed)',
        f'API available: {"✅ Yes" if api_available else "❌ No (local extraction)"}',
        f'Average query time: {avg_query_time:.1f}s',
        f'Router chose Hybrid: {hybrid_used}/{len(results)} ({hybrid_used/len(results)*100:.1f}%)',
        f'Router chose Standard: {standard_used}/{len(results)} ({standard_used/len(results)*100:.1f}%)',
        f'Token savings: {token_savings:,} ({token_savings_pct:.1f}%)',
        '',
        '| Question ID | Type | Std Latency | Std Tokens | Std Relevance | Hybrid Latency | Hybrid Tokens | Hybrid Relevance | Router | API |',
        '|-------------|------|-------------|------------|---------------|----------------|---------------|------------------|--------|-----|'
    ]
    for r in results:
        api_status = '✅' if r['api_used'] else '❌'
        lines.append(
            f"| {r['question_id']} | {r['question_type']} | "
            f"{r['std_latency']:.2f}s | {r['std_tokens']} | {r['std_relevance']:.1f}% | "
            f"{r['hyb_latency']:.2f}s | {r['hyb_tokens']} | {r['hyb_relevance']:.1f}% | "
            f"{r['router_chosen']} | {api_status} |"
        )

    avg_std_rel = sum(r['std_relevance'] for r in results) / len(results)
    avg_hyb_rel = sum(r['hyb_relevance'] for r in results) / len(results)
    lines.extend([
        '',
        '## Summary Statistics',
        f'- **Average Standard relevance:** {avg_std_rel:.1f}%',
        f'- **Average Hybrid relevance:** {avg_hyb_rel:.1f}%',
        f'- **Improvement:** {avg_hyb_rel - avg_std_rel:.1f} percentage points',
        f'- **Total token savings:** {token_savings:,} ({token_savings_pct:.1f}%)',
        f'- **Average query latency:** {avg_query_time:.1f}s',
        f'- **Router chose Hybrid:** {hybrid_used}/{len(results)} ({hybrid_used/len(results)*100:.1f}%)',
    ])
    Path(OUTPUT_MD).write_text('\n'.join(lines), encoding='utf-8')
    print(f'📄 Markdown report saved to {OUTPUT_MD}')
else:
    print('No results to save.')