"""
Gradio web UI for Hybrid RAG.
Users paste their OpenAI API key, type a query, and get answers.
The database lives server-side — users never download anything.
"""

import os
import gradio as gr
from v24 import build_large_corpus_engine

print("Loading Hybrid RAG system (1.9M chunks, 34M entities)...")
db, processor, engine = build_large_corpus_engine()
print("System ready.")


def search(query: str, api_key: str, compare_mode: bool):
    """Run a query and return formatted results."""
    if not query.strip():
        return "Please enter a query.", ""
    if not api_key.strip():
        return "Please paste your OpenAI API key.", ""
    if not api_key.strip().startswith("sk-"):
        return "Invalid API key format. Should start with 'sk-'.", ""

    key = api_key.strip()
    os.environ["OPENAI_API_KEY"] = key  # fallback for engine internals

    try:
        if compare_mode:
            # True side-by-side: both pipelines run unconditionally,
            # no third direct-AI call.
            results = engine.compare(
                query, provider="auto", api_key=key,
                include_direct_ai=False, force_hybrid=True,
            )
        else:
            # Single forced-Hybrid query (full quality path).
            results = {
                "hybrid_rag": engine.compare_single(
                    query, method="hybrid_rag", provider="auto", api_key=key,
                )
            }

        labels = {
            "hybrid_rag": "Hybrid RAG",
            "standard_rag": "Standard RAG",
            "direct_ai": "Direct AI (no retrieval)",
        }

        output_parts = []
        for method, data in results.items():
            label = labels.get(method, method)
            answer = data.get("answer", "No answer generated.")
            confidence = data.get("confidence_score", 0) or 0
            latency = data.get("latency", 0) or 0
            tokens = data.get("tokens_used", 0) or 0
            chunks = data.get("chunks_retrieved", 0) or 0
            savings = data.get("token_savings_percent_vs_standard")

            stats = (
                f"📊 **Confidence:** {confidence:.1%} | "
                f"⏱️ **Latency:** {latency:.1f}s | "
                f"**Tokens:** {tokens} | "
                f"**Chunks:** {chunks}"
            )
            if savings is not None:
                stats += f" | **Token savings vs Standard:** {savings:.1f}%"

            output_parts.append(f"## {label}\n\n{answer}\n\n---\n{stats}")

        sources = ""
        for method, data in results.items():
            label = labels.get(method, method)
            source_list = data.get("sources", []) or []
            if source_list:
                sources += f"### {label} — Retrieved from:\n"
                for src in source_list:
                    sources += f"- {src}\n"
                sources += "\n"

        return "\n\n---\n\n".join(output_parts), sources

    except Exception as e:
        return f"Error: {str(e)}", ""


def run_eval(num_questions: int, api_key: str):
    """Run a quick evaluation and return summary stats."""
    if not api_key.strip().startswith("sk-"):
        return "Please paste a valid OpenAI API key."

    os.environ["OPENAI_API_KEY"] = api_key.strip()

    try:
        import subprocess
        result = subprocess.run(
            ["python", "batch_compare_v2.py", "--force-hybrid",
             "--sample", str(int(num_questions)), "--output", "web_eval"],
            capture_output=True, text=True, timeout=600, cwd="/app"
        )
        return result.stdout if result.returncode == 0 else result.stderr
    except subprocess.TimeoutExpired:
        return "Evaluation timed out. Try fewer questions."
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="Hybrid RAG — Enterprise Search") as app:

    gr.HTML("""
    <div style="text-align:center; margin-bottom:20px;">
    <h1>🔍 Hybrid RAG — Multi-Source Enterprise Search</h1>
    <p>Search across 1.9M chunks from 9 enterprise data sources (Slack, Gmail, GitHub, Jira, Confluence, HubSpot, Linear, Fireflies, Google Drive).</p>
    <p>Entity knowledge graph with 34M rows • FTS5 + RRF + Cross-encoder reranking</p>
    </div>
    """)

    with gr.Tab("Search"):
        api_input = gr.Textbox(
            label="OpenAI API Key",
            placeholder="sk-...",
            type="password",
            info="Your key is used only for this request — it's not stored."
        )

        query_input = gr.Textbox(
            label="Query",
            placeholder="e.g., What GitHub PRs relate to the Proxima Bank 429 incident?",
            lines=2
        )

        with gr.Row():
            compare_checkbox = gr.Checkbox(
                label="Compare Standard vs Hybrid side-by-side (runs both pipelines)",
                value=False
            )
            search_btn = gr.Button("Search", variant="primary")

        answer_output = gr.Markdown(label="Answer")
        sources_output = gr.Markdown(label="Sources")

        search_btn.click(
            fn=search,
            inputs=[query_input, api_input, compare_checkbox],
            outputs=[answer_output, sources_output]
        )

    with gr.Tab("Evaluate"):
        gr.Markdown("""
        ### Quick Evaluation
        Run a sample of questions to see Standard vs Hybrid performance.
        This uses your API key and may take several minutes.
        """)
        eval_api = gr.Textbox(
            label="OpenAI API Key",
            placeholder="sk-...",
            type="password"
        )
        num_q = gr.Slider(
            minimum=5, maximum=50, value=10, step=5,
            label="Number of questions"
        )
        eval_btn = gr.Button("Run Evaluation", variant="primary")
        eval_output = gr.Textbox(label="Results", lines=15)

        eval_btn.click(
            fn=run_eval,
            inputs=[num_q, eval_api],
            outputs=[eval_output]
        )

    with gr.Tab("About"):
        gr.Markdown("""
        ## Architecture

        **Standard RAG:** Query → FTS5 (BM25) → Top 5 chunks → LLM → Answer

        **Hybrid RAG:** Query → spaCy NER → Partition Routing → FTS5 + Entity Graph → RRF Fusion → Cross-encoder Reranking → LLM → Answer

        ## Results (500-question evaluation)

        | Metric | Standard | Hybrid | Difference |
        |--------|----------|--------|------------|
        | Answer relevance | 65.3% | 66.1% | +0.9 pp |
        | Token usage | 459,014 | 423,874 | -7.7% |
        | Win rate | 36% | 39% | +3 pp |

        ## Dataset
        - 1,924,473 chunks across 9 enterprise data sources
        - 34M entity rows in the knowledge graph
        - 500 evaluation questions across 10 question types

        ## Privacy
        Your OpenAI API key is sent directly to OpenAI's API from this server.
        It is not stored, logged, or shared. Only the query and retrieved chunks
        are sent to OpenAI for answer generation.
        """)

app.launch(server_name="0.0.0.0", server_port=7860)
