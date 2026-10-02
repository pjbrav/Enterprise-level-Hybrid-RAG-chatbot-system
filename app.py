"""Gradio UI for Hybrid RAG.

Tabs: Search | My library | Host | Join | Evaluate | About.
All behaviour lives in rag.app_controller.AppController; this file is layout and wiring.

The admin UI binds to 127.0.0.1 by default. Only the read-only host API (a separate
port, started from the Host tab) is reachable from the network.
"""

from __future__ import annotations

import os

import gradio as gr

from rag.app_controller import (MODE_PRODUCTION, MODE_TESTING, PROVIDER_LABELS, AppController, ProviderSettings)

DEFAULT_HOST_PORT = int(os.getenv("HOST_PORT", "8765"))
LOCAL_PROVIDER_LABELS = {"Ollama (local)"}
URL_PROVIDER_LABELS = {"Ollama (local)", "OpenAI-compatible"}

ABOUT = """
## How it works

**Standard RAG:** question -> keyword search -> top chunks -> LLM.

**Hybrid RAG:** question -> entity extraction + partition routing -> keyword search **and** entity graph per partition
-> rank fusion -> (rarely) cross-encoder rerank -> LLM. It reaches the right evidence with fewer tokens.

**Hybrid intensity** (0 -> 1) controls how much work Hybrid does: candidate pool size, how many partitions are searched
and when the rerank step kicks in. 0.5 is the configuration the published evaluation used.

## Sharing a corpus
Anyone can **host** a corpus from their own machine: issue a one-time code, share the URL and code, and members query it
remotely. Members never download the files or the database; they receive only the top few chunks for each question,
held in memory for that request. Answers are generated on the member's machine with the member's own API key.
The host can **kick** a member at any time and their access stops immediately.

## Privacy
- API keys you type are kept in memory for the request only; they are not written to disk or to environment variables.
- Only the question and the retrieved chunks are sent to the LLM provider you choose.
- Host/member traffic is plain HTTP: use a trusted network or a tunnel. The one-time code is the only secret.
- Relevance numbers are lexical overlap *proxies*, not accuracy scores.
"""


def _settings(label, key, model, base_url) -> ProviderSettings:
    return ProviderSettings(PROVIDER_LABELS.get(label, "openai"), (key or "").strip(), (model or "").strip(),
                            (base_url or "").strip())


def build_app(controller: AppController | None = None) -> gr.Blocks:
    ctl = controller or AppController()

    with gr.Blocks(title="Hybrid RAG") as demo:
        gr.Markdown("# Hybrid RAG\nAsk questions over your own documents or a corpus someone shares with you, "
                    "using fewer tokens than standard RAG.")

        # ---- provider bar (shared by Search / Evaluate) ---------------------------
        with gr.Row():
            provider = gr.Dropdown(list(PROVIDER_LABELS), value="OpenAI", label="LLM provider", scale=2)
            model = gr.Textbox(label="Model (optional)", placeholder="provider default", scale=2)
            api_key = gr.Textbox(label="API key", type="password", placeholder="kept in memory only; never saved", scale=3)
            base_url = gr.Textbox(label="Base URL", placeholder="http://localhost:11434/v1", visible=False, scale=3)
        with gr.Row():
            test_btn = gr.Button("Test connection", size="sm", scale=0)
            test_status = gr.Markdown("")

        def on_provider(label):
            return (gr.Textbox(visible=label not in LOCAL_PROVIDER_LABELS),
                    gr.Textbox(visible=label in URL_PROVIDER_LABELS))

        provider.change(on_provider, provider, [api_key, base_url])
        test_btn.click(lambda p, k, m, b: ctl.test_connection(_settings(p, k, m, b)),
                       [provider, api_key, model, base_url], test_status)

        # Created here (rendered inside the Host tab) so the Library handlers can refresh it.
        h_corpus = gr.Dropdown(ctl.corpus_choices(), label="Corpus to host", scale=3, render=False)

        with gr.Tabs():
            # ======================= Search =======================
            with gr.Tab("Search"):
                sources = gr.CheckboxGroup(label="Search in", choices=ctl.source_choices(), value=[])
                refresh_sources = gr.Button("Refresh list", size="sm")
                question = gr.Textbox(label="Question", lines=2, placeholder="Ask anything about the selected corpora")
                with gr.Row():
                    mode = gr.Radio([MODE_PRODUCTION, MODE_TESTING], value=MODE_PRODUCTION, label="Mode", scale=2)
                    intensity = gr.Slider(0.0, 1.0, value=0.5, step=0.05, scale=3,
                                          label="Hybrid intensity (faster <- 0 ... 1 -> more thorough)")
                run_btn = gr.Button("Search", variant="primary")
                banner = gr.Markdown("")
                with gr.Row(equal_height=False):
                    standard_out = gr.Markdown("", label="Standard RAG")
                    hybrid_out = gr.Markdown("", label="Hybrid RAG")

                def do_search(q, srcs, md, inten, p, k, m, b):
                    res = ctl.search(q, srcs, md, inten, _settings(p, k, m, b))
                    if res.get("error"):
                        return res["error"], "", ""
                    standard, hybrid = res["standard"], res["hybrid"]
                    savings = ctl.token_savings(standard, hybrid)
                    std_md = ctl.render_result("Standard RAG", standard) if standard else \
                        "### Standard RAG\n\n_Not run in Production mode. Switch to Testing to compare._"
                    return "", std_md, ctl.render_result("Hybrid RAG", hybrid, savings, res.get("no_key", False))

                run_btn.click(do_search, [question, sources, mode, intensity, provider, api_key, model, base_url],
                              [banner, standard_out, hybrid_out])
                question.submit(do_search, [question, sources, mode, intensity, provider, api_key, model, base_url],
                                [banner, standard_out, hybrid_out])
                refresh_sources.click(lambda cur: gr.CheckboxGroup(choices=ctl.source_choices(), value=cur), sources, sources)

            # ======================= My library =======================
            with gr.Tab("My library"):
                gr.Markdown("Upload documents for **this machine only**. They are indexed locally and are separate from "
                            "anything you host or join.")
                with gr.Row():
                    lib_name = gr.Textbox(label="Corpus name", value="My documents", scale=2)
                    lib_retain = gr.Checkbox(label="Keep after I close the app", value=True, scale=1)
                lib_files = gr.File(label="Files (pdf, txt, md, py, js, ipynb, or a zip)", file_count="multiple",
                                    type="filepath")
                lib_btn = gr.Button("Index documents", variant="primary")
                lib_msg = gr.Markdown("")
                gr.Markdown("### Corpora on this machine")
                lib_table = gr.Dataframe(headers=["Name", "Role", "Backend", "Mode", "Retention", "Docs", "Chunks",
                                                  "Partitions", "id"], value=ctl.corpus_rows(), interactive=False,
                                         wrap=True)
                with gr.Row():
                    manage_corpus = gr.Dropdown(ctl.corpus_choices(), label="Manage corpus", scale=3)
                    lib_refresh = gr.Button("Refresh", size="sm", scale=1)
                    lib_delete = gr.Button("Delete corpus", variant="stop", size="sm", scale=1)
                manage_msg = gr.Markdown("")
                with gr.Accordion("Routing keywords (partition: keyword, keyword, ...)", open=False):
                    kw_text = gr.Textbox(lines=8, label="Keywords used to route a question to partitions")
                    with gr.Row():
                        kw_save = gr.Button("Save keywords", size="sm")
                        kw_regen = gr.Button("Regenerate (discards edits)", size="sm")
                with gr.Accordion("Attach an existing SQLite database (read-only)", open=False):
                    gr.Markdown("For a Hybrid RAG database you already built (for example `data/hybrid_rag_v22.db`). "
                                "It is opened read-only and never copied or modified.")
                    with gr.Row():
                        att_path = gr.Textbox(label="Path to .db file", scale=3)
                        att_name = gr.Textbox(label="Name", scale=1)
                    att_btn = gr.Button("Attach", size="sm")
                    att_msg = gr.Markdown("")

                def refresh_library(current=None):
                    """table, manage dropdown, search checklist, host dropdown"""
                    return (ctl.corpus_rows(), gr.Dropdown(choices=ctl.corpus_choices(), value=current),
                            gr.CheckboxGroup(choices=ctl.source_choices()), gr.Dropdown(choices=ctl.corpus_choices()))

                lib_outputs = [lib_table, manage_corpus, sources, h_corpus]

                def do_index(name, files, retain):
                    msg = ctl.add_documents(name, files or [], retain)
                    return (msg, *refresh_library())

                lib_btn.click(do_index, [lib_name, lib_files, lib_retain], [lib_msg, *lib_outputs])
                lib_refresh.click(lambda c: refresh_library(c), manage_corpus, lib_outputs)

                def do_delete(cid):
                    msg = ctl.delete_corpus(cid)
                    return (msg, *refresh_library())

                lib_delete.click(do_delete, manage_corpus, [manage_msg, *lib_outputs])
                manage_corpus.change(lambda c: ctl.keywords_text(c), manage_corpus, kw_text)
                kw_save.click(lambda c, t: ctl.save_keywords(c, t), [manage_corpus, kw_text], manage_msg)
                kw_regen.click(lambda c: (ctl.regenerate_keywords(c), ctl.keywords_text(c)), manage_corpus, [manage_msg, kw_text])

                def do_attach(path, name):
                    msg = ctl.attach_database(name, path)
                    return (msg, *refresh_library())

                att_btn.click(do_attach, [att_path, att_name], [att_msg, *lib_outputs])

            # ======================= Host =======================
            with gr.Tab("Host"):
                gr.Markdown("Share one of your corpora. Members search it from their own machine without ever receiving "
                            "your files or database. **Only you** can add, replace or delete what is hosted (under *My library*).")
                with gr.Row():
                    h_corpus.render()
                    h_port = gr.Number(value=DEFAULT_HOST_PORT, precision=0, label="Port", scale=1)
                with gr.Row():
                    h_start = gr.Button("Start hosting", variant="primary")
                    h_stop = gr.Button("Stop hosting")
                    h_remove = gr.Checkbox(label="...and remove all members", value=False)
                h_status = gr.Markdown("Not hosting.")
                with gr.Row():
                    h_ttl = gr.Slider(5, 240, value=30, step=5, label="Code valid for (minutes)")
                    h_uses = gr.Number(value=10, precision=0, label="Max joins per code")
                with gr.Row():
                    h_issue = gr.Button("Issue one-time code")
                    h_rotate = gr.Button("Rotate (invalidate old codes)")
                h_code = gr.Markdown("")
                h_alert = gr.Markdown("")
                gr.Markdown("### Members")
                h_members = gr.Dataframe(headers=["id", "Name", "Address", "Joined", "Last seen"], interactive=False)
                with gr.Row():
                    h_kick_id = gr.Number(label="Member id to remove", precision=0, scale=2)
                    h_kick = gr.Button("Kick", variant="stop", scale=1)
                h_kick_msg = gr.Markdown("")
                h_timer = gr.Timer(5)

                def refresh_host(cid):
                    return ctl.members_rows(cid), ctl.security_banner(cid), ctl.host_status(cid)

                h_corpus.change(refresh_host, h_corpus, [h_members, h_alert, h_status])
                h_timer.tick(lambda cid: refresh_host(cid)[:2], h_corpus, [h_members, h_alert])
                h_start.click(lambda cid, port: ctl.start_hosting(cid, port), [h_corpus, h_port], h_status)
                h_stop.click(lambda cid, rm: ctl.stop_hosting(cid, rm), [h_corpus, h_remove], h_status)
                h_issue.click(lambda cid, ttl, n: ctl.issue_otp(cid, ttl, n), [h_corpus, h_ttl, h_uses], h_code)
                h_rotate.click(lambda cid, ttl, n: ctl.issue_otp(cid, ttl, n, rotate=True), [h_corpus, h_ttl, h_uses], h_code)
                h_kick.click(lambda mid, cid: (ctl.kick(mid), ctl.members_rows(cid)), [h_kick_id, h_corpus], [h_kick_msg, h_members])

            # ======================= Join =======================
            with gr.Tab("Join"):
                gr.Markdown("Join a corpus someone else is hosting. **Nothing is downloaded**: only a membership token is "
                            "stored on this machine, and the host can remove you at any time.")
                with gr.Row():
                    j_url = gr.Textbox(label="Host URL", placeholder="http://192.168.1.20:8765", scale=3)
                    j_otp = gr.Textbox(label="One-time code", placeholder="6 digits", scale=1)
                    j_name = gr.Textbox(label="Your name", value="member", scale=2)
                j_btn = gr.Button("Join", variant="primary")
                j_msg = gr.Markdown("")
                gr.Markdown("### Corpora you joined")
                j_table = gr.Dataframe(headers=["id", "Corpus", "Host", "Your name", "Status"], value=ctl.joins_rows(),
                                       interactive=False, wrap=True)
                with gr.Row():
                    j_leave_id = gr.Number(label="id to leave", precision=0, scale=2)
                    j_leave = gr.Button("Leave", scale=1)

                def do_join(url, otp, name):
                    msg = ctl.join(url, otp, name)
                    return msg, ctl.joins_rows(), gr.CheckboxGroup(choices=ctl.source_choices())

                def do_leave(jid):
                    msg = ctl.leave(jid)
                    return msg, ctl.joins_rows(), gr.CheckboxGroup(choices=ctl.source_choices())

                j_btn.click(do_join, [j_url, j_otp, j_name], [j_msg, j_table, sources])
                j_leave.click(do_leave, j_leave_id, [j_msg, j_table, sources])

            # ======================= Evaluate =======================
            with gr.Tab("Evaluate"):
                gr.Markdown("Runs a sample of the evaluation questions against `data/hybrid_rag_v22.db` (Standard vs Hybrid) "
                            "using the provider selected above. This uses your API key and can take several minutes.")
                e_n = gr.Slider(5, 50, value=10, step=5, label="Number of questions")
                e_btn = gr.Button("Run evaluation", variant="primary")
                e_out = gr.Textbox(label="Results", lines=15)
                e_btn.click(lambda n, p, k, m, b: ctl.run_eval(n, _settings(p, k, m, b)),
                            [e_n, provider, api_key, model, base_url], e_out)

            with gr.Tab("About"):
                gr.Markdown(ABOUT)

        # Populate everything fresh whenever the page is (re)loaded.
        demo.load(lambda: (gr.CheckboxGroup(choices=ctl.source_choices()), ctl.corpus_rows(),
                           gr.Dropdown(choices=ctl.corpus_choices()), gr.Dropdown(choices=ctl.corpus_choices()),
                           ctl.joins_rows()),
                  None, [sources, lib_table, manage_corpus, h_corpus, j_table])
    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name=os.getenv("GRADIO_BIND", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        theme=gr.themes.Soft(),
    )
