import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import streamlit as st

from finrag.documents import load_demo, parse_bytes
from finrag.engine import PipelineConfig, RAGIndex
from finrag.evaluation import evaluate, presets
from finrag.generation import DEFAULT_MODEL, generate, make_client, resolve_model


st.set_page_config(page_title="FinCon | John Khamis", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root {
  --bg: rgb(10,17,28); --bg-side: rgb(14,24,38); --bg-card: rgb(16,29,43);
  --border: rgb(35,52,68); --text: rgb(228,237,244); --muted: rgb(155,176,194);
  --accent: rgb(118,226,194); --link: rgb(108,168,237);
  color-scheme: dark;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background: var(--bg) !important; color: var(--text) !important;}
[data-testid="stHeader"], [data-testid="stToolbar"] {background: var(--bg) !important; color: var(--text) !important;}
[data-testid="stHeader"] *, [data-testid="stToolbar"] * {color: var(--muted) !important; fill: var(--muted) !important;}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {background: var(--bg-side) !important; border-right: 1px solid rgb(35,49,65);}
[data-testid="stSidebar"] *:not(input):not(textarea):not([data-baseweb="select"] *):not([data-baseweb="popover"] *) {color: var(--text);}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color: var(--muted) !important;}
[data-testid="stSidebar"] hr {border-color: rgb(44,66,78) !important;}
h1, h2, h3, h4, h5, h6, p, li, label, span, [data-testid="stMarkdownContainer"] {color: inherit;}
h1, h2, h3, h4 {color: var(--text) !important;}
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {color: var(--muted) !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {color: var(--muted) !important;}
a, a:visited {color: var(--link) !important;}
.block-container {padding-top: 4rem; padding-bottom: 3rem; max-width: 1350px;}
h1 {letter-spacing: -0.045em; font-weight: 650 !important;}
h2,h3 {letter-spacing: -0.025em;}
.eyebrow {color: var(--accent) !important; font-size: 0.78rem; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.6rem;}
.hero {padding: 1.5rem 0 1.2rem; max-width: 900px;}
.hero p {font-size: 1.03rem; color: var(--muted) !important; line-height: 1.8;}
.pipeline {padding: 1rem 1.2rem; border: 1px solid rgb(44,66,78); border-radius: 12px; color: rgb(148,219,201) !important; background: rgb(15,30,41); margin-bottom: 1.2rem; font-size: 0.88rem;}
.quiet {color: rgb(144,165,184) !important; font-size: 0.88rem; line-height: 1.7;}
[data-testid="stMetric"] {background: var(--bg-card) !important; border: 1px solid var(--border); border-radius: 12px; padding: 1rem;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {color: var(--muted) !important;}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {font-size: 1.65rem; color: var(--text) !important;}
/* inputs */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="textarea"], .stTextArea textarea, .stTextInput input {background: rgb(24,38,55) !important; color: var(--text) !important; border-color: rgb(52,72,92) !important;}
[data-baseweb="select"] *, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {color: var(--text) !important;}
[data-baseweb="select"] svg {fill: var(--muted) !important;}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul, [data-baseweb="menu"] {background: rgb(20,33,48) !important;}
[data-baseweb="popover"] li, [data-baseweb="popover"] li * {color: var(--text) !important; background: transparent;}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] {background: rgb(34,54,72) !important;}
::placeholder {color: rgb(110,132,152) !important; opacity: 1 !important;}
/* expanders, tabs, uploader */
[data-testid="stExpander"] {background: transparent; border: 1px solid rgb(44,66,78) !important; border-radius: 10px;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {color: var(--text) !important;}
[data-testid="stFileUploaderDropzone"] {background: rgb(20,33,48) !important;}
[data-testid="stFileUploaderDropzone"] * {color: var(--muted) !important;}
[data-baseweb="tab"] {color: var(--muted) !important;}
[data-baseweb="tab"][aria-selected="true"] {color: var(--accent) !important;}
[data-baseweb="tab-highlight"] {background-color: var(--accent) !important;}
[data-baseweb="tab-border"] {background-color: rgb(44,66,78) !important;}
[data-testid="stAlert"] * {color: var(--text) !important;}
[data-testid="stDataFrame"] {background: var(--bg-card);}
code {color: rgb(148,219,201) !important; background: rgb(20,33,48) !important;}
.stButton button {border-radius: 9px; color: var(--text); background: rgb(24,38,55); border: 1px solid rgb(52,72,92);}
.stButton button:hover {border-color: var(--accent); color: var(--accent);}
.stButton button[kind="primary"] {background: var(--accent); color: rgb(10,17,28); border-color: var(--accent);}
.stTextArea textarea {line-height: 1.7;}
/* never dim stale elements while the app reruns */
[data-stale="true"], .stale-element {opacity: 1 !important; transition: none !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def demo_index(method):
    return RAGIndex(load_demo(), method)


def secret(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.getenv(name, default)


def example(question):
    st.session_state["question"] = question


def show_result(result, compact=False):
    for warning in result["warnings"]:
        st.warning(warning)
    if result.get("generation_warning"):
        st.warning(result["generation_warning"])
    st.caption(result.get("answer_mode", "Retrieval only") + " · " + result["route"]["domain"])
    if "answer" in result:
        st.markdown(result["answer"])
    if not compact:
        st.markdown("**Evidence trail**")
    for hit in result["hits"]:
        page = f" · page {hit['page']}" if hit["page"] else ""
        with st.expander(f"[{hit['citation']}] {hit['title']}{page}"):
            st.caption(f"{hit['source']} · {hit['domain']} · record {hit['document_id']}")
            st.write(hit["text"])
            st.caption(f"Retrieval score: {hit['retrieval_score']:.3f} · Rerank score: {hit['rerank_score']:.3f}")
    if not compact:
        with st.expander("Inspect the pipeline"):
            st.markdown("**Search variants**")
            for query in result["search_queries"]:
                st.write(query)
            if result["feedback_terms"]:
                st.write("Feedback terms: " + ", ".join(result["feedback_terms"]))
            st.caption("Effective expansion: " + result["effective_expansion"])
            st.json(result["route"])
            st.json(result["config"])
            st.caption(result.get("citation_check", ""))
        st.download_button("Download answer & trace", json.dumps(result, ensure_ascii=False, indent=2), "fincon-answer.json", "application/json", key="download_answer")


with st.sidebar:
    st.subheader("◈ FINCON")
    st.caption("RETRIEVAL RESEARCH LAB · TASK 02")
    st.caption("John Khamis · Project owner")
    st.divider()
    st.markdown("**Build your pipeline**")
    chunking = st.selectbox("Document chunking", ["sentence_window", "semantic"], format_func=lambda v: {"sentence_window": "Overlapping sentence windows", "semantic": "Lexical cohesion boundaries"}[v])
    expansion = st.selectbox("Query expansion", ["prf", "rocchio", "multi_query", "hyde", "none"], format_func=lambda v: {"prf": "Pseudo-relevance feedback", "rocchio": "Rocchio vector feedback", "multi_query": "LLM multi-query · needs Groq", "hyde": "HyDE · needs Groq", "none": "Original question"}[v])
    routing = st.selectbox("Query routing", ["centroid", "knn", "none"], format_func=lambda v: {"centroid": "Domain centroid", "knn": "Nearest-neighbor vote", "none": "Search all domains"}[v])
    with st.expander("Retrieval & context"):
        retrieval = st.selectbox("Retrieval method", ["hybrid", "bm25", "lsa"])
        fusion = st.selectbox("Fusion method", ["relative_score", "rrf"])
        reranking = st.selectbox("Reranking method", ["mmr", "relevance"])
        context = st.selectbox("Context assembly", ["compressed", "window"])
        top_k = st.slider("Evidence chunks", 1, 8, 4)
        dense_weight = st.slider("LSA weight in hybrid search", 0.0, 1.0, 0.55, 0.05)
        mmr_lambda = st.slider("MMR relevance weight", 0.0, 1.0, 0.75, 0.05)
    with st.expander("Answer generation"):
        st.caption("Without a key, the app returns source excerpts. Groq enables synthesized answers, multi-query and HyDE.")
        api_key = st.text_input("Groq API key", type="password", help="Session only. For deployment, use Streamlit Secrets.")
        configured_model = secret("GROQ_MODEL", DEFAULT_MODEL)
        model = st.text_input("Groq model", value=resolve_model(configured_model))
        if configured_model.strip() != resolve_model(configured_model):
            st.caption("A retired model setting was migrated to " + resolve_model(configured_model) + ".")
    with st.expander("Your documents"):
        files = st.file_uploader("Add PDF, TXT, MD or JSON", type=["pdf", "txt", "md", "json"], accept_multiple_files=True, help="10 MB per file, 20 files and 1,500,000 text characters per session. Text PDFs only.")
        corpus_mode = st.radio("Collection", ["Demo corpus", "Demo + uploads", "Uploads only"])
        st.caption("Uploads stay in this server session. When Groq is enabled, selected passages and your question are sent to Groq.")
    st.divider()
    st.caption("English + limited Arabic search glossary. Groq can answer in Arabic. Local vectors use LSA, not a pretrained neural embedding model.")

config = PipelineConfig(chunking, expansion, routing, retrieval, fusion, reranking, context, top_k, dense_weight, mmr_lambda)
llm = make_client(api_key or secret("GROQ_API_KEY"), model)
try:
    if corpus_mode == "Demo corpus":
        index = demo_index(chunking)
        collection_signature = "demo:" + chunking
    else:
        if not files:
            st.info("Upload at least one document or select Demo corpus.")
            st.stop()
        if len(files) > 20:
            raise ValueError("Upload at most 20 files per session.")
        collection_signature = hashlib.sha256((chunking + corpus_mode).encode() + b"".join(f.name.encode() + f.getvalue() for f in files)).hexdigest()
        if st.session_state.get("collection_signature") != collection_signature:
            documents = load_demo() if corpus_mode == "Demo + uploads" else []
            for number, file in enumerate(files):
                documents.extend(replace(d, id=f"upload-{number}-{d.id}") for d in parse_bytes(file.name, file.getvalue()))
            if sum(len(d.text) for d in documents) > 1_500_000:
                raise ValueError("This collection exceeds the 1,500,000-character limit.")
            st.session_state["uploaded_index"] = RAGIndex(documents, chunking)
            st.session_state["collection_signature"] = collection_signature
        index = st.session_state["uploaded_index"]
except Exception as error:
    st.error("Could not build the collection. " + str(error)[:240])
    st.stop()

st.markdown('<div class="eyebrow">Financial intelligence / source by source</div>', unsafe_allow_html=True)
st.title("John Khamis | Financial RAG Lab")
st.caption("Project owner: John Khamis · [GitHub](https://github.com/Yassahany2004)")
st.markdown('<div class="hero"><p>Explore financial documents with a RAG pipeline you can inspect, change and measure.<br>اسأل، غيّر طرق البحث، وشوف الإجابة جاية منين.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="pipeline">01 &nbsp; CHUNK &nbsp; → &nbsp; 02 &nbsp; EXPAND & ROUTE &nbsp; → &nbsp; 03 &nbsp; RETRIEVE &nbsp; → &nbsp; 04 &nbsp; RERANK &nbsp; → &nbsp; 05 &nbsp; CITE</div>', unsafe_allow_html=True)
metrics = st.columns(4)
metrics[0].metric("Documents", len(index.documents))
metrics[1].metric("Searchable chunks", len(index.chunks))
metrics[2].metric("Knowledge domains", len(index.domains))
metrics[3].metric("Answer mode", "Groq" if llm else "Excerpts")
st.caption("Demo records are course samples from Mile-1; their regulatory dates and claims have not been verified against current law.")

ask_tab, compare_tab, benchmark_tab, about_tab = st.tabs(["Ask the collection", "Compare methods", "Benchmark", "How it works"])

with ask_tab:
    st.subheader("Follow the evidence")
    examples = st.columns(3)
    examples[0].button("Cyber incident deadline", on_click=example, args=("When must a material cybersecurity incident be disclosed under Form 8-K?",), width="stretch")
    examples[1].button("Scope 1, 2 & 3 emissions", on_click=example, args=("How do Scope 1, Scope 2 and Scope 3 emissions differ?",), width="stretch")
    examples[2].button("اسأل بالعربي", on_click=example, args=("ايه الفرق بين انبعاثات النطاق 1 و2 و3؟",), width="stretch")
    with st.form("question_form"):
        question = st.text_area("Your question / سؤالك", key="question", placeholder="For example: How does APV differ from WACC when a company's debt changes?", height=110, max_chars=2000)
        submitted = st.form_submit_button("Retrieve & answer →", type="primary")
    if submitted:
        if not question.strip():
            st.warning("Write a question first / اكتب سؤالك الأول.")
        else:
            with st.spinner("Finding evidence and assembling the answer…"):
                try:
                    st.session_state["result"] = generate(index.retrieve(question, config, llm), llm)
                    st.session_state["result_collection"] = collection_signature
                except Exception:
                    st.error("The query could not be processed. Try a shorter or more specific question.")
    result = st.session_state.get("result")
    if result and st.session_state.get("result_collection") == collection_signature:
        if result["config"] != asdict(config):
            st.info("Settings changed. Submit again to apply them to this answer.")
        st.divider()
        st.caption(f"RETRIEVAL {result['retrieval_ms']:.1f} ms · {len(result['hits'])} EVIDENCE CHUNKS")
        show_result(result)
    else:
        st.markdown('<p class="quiet">Start with a sample or ask your own question. Every answer includes the passages used to produce it.</p>', unsafe_allow_html=True)

with compare_tab:
    st.subheader("Same question. Different methods.")
    st.write("Compare a BM25L baseline with your current settings. Both use the active collection. This view skips answer generation; LLM query expansion can still call Groq when selected.")
    compare_question = st.text_input("Comparison question", value="How do WACC and APV account for the tax benefits of debt?")
    if st.button("Compare pipelines"):
        if not compare_question.strip() or len(compare_question) > 2000:
            st.warning("Enter a question between 1 and 2000 characters.")
        else:
            with st.spinner("Running both retrieval pipelines…"):
                baseline = replace(presets()["BM25L baseline"], chunking=chunking, top_k=top_k)
                left = index.retrieve(compare_question, baseline)
                right = index.retrieve(compare_question, config, llm)
                st.session_state["comparison"] = (left, right, collection_signature)
    comparison = st.session_state.get("comparison")
    if comparison and comparison[2] == collection_signature:
        for column, label, result in zip(st.columns(2), ["BM25L baseline", "Your pipeline · at time of comparison"], comparison[:2]):
            with column:
                st.markdown("**" + label + "**")
                st.caption(f"{result['retrieval_ms']:.1f} ms · {len(result['hits'])} chunks")
                if not result["hits"]:
                    st.info("No supporting passages found.")
                show_result(result, compact=True)

with benchmark_tab:
    st.subheader("Measure before you choose")
    st.write("Four local configurations, 14 labeled questions and record-level retrieval metrics. The benchmark always uses the bundled demo corpus, independently of uploads or sidebar settings.")
    st.caption("Hit@4: labeled record appears in up to four retrieved chunks. MRR@4: reciprocal position among deduplicated retrieved records. Evidence-term coverage: fraction of labeled strings found in the assembled context. These do not measure answer faithfulness.")
    if st.button("Run retrieval benchmark"):
        with st.spinner("Evaluating 56 retrieval runs…"):
            st.session_state["benchmark"] = evaluate()
    report = st.session_state.get("benchmark")
    if report:
        table = pd.DataFrame(report["summary"])
        st.dataframe(table, hide_index=True, width="stretch")
        st.bar_chart(table.set_index("pipeline")[["hit_at_4", "mrr_at_4", "evidence_term_coverage"]], color=["rgb(118,226,194)", "rgb(108,168,237)", "rgb(195,167,242)"], stack=False)
        st.caption(report["description"])
        with st.expander("Inspect every test case"):
            st.dataframe(pd.DataFrame(report["details"]), hide_index=True, width="stretch")
        st.download_button("Download evaluation report", json.dumps(report, ensure_ascii=False, indent=2), "evaluation-report.json", "application/json")

with about_tab:
    st.subheader("A different method at every stage")
    st.dataframe(pd.DataFrame([
        {"Stage": "Chunking", "Mile-1": "Fixed parent/child character chunks", "This project": "Overlapping sentences or lexical cohesion boundaries"},
        {"Stage": "Expansion", "Mile-1": "spaCy + financial synonym dictionary", "This project": "PRF, Rocchio, LLM multi-query, HyDE"},
        {"Stage": "Routing", "Mile-1": "No router found in the provided code", "This project": "Centroid similarity or KNN vote; soft domain boost"},
        {"Stage": "Retrieval", "Mile-1": "MiniLM vectors + ChromaDB + BM25", "This project": "BM25L + local TF-IDF → truncated SVD → LSA cosine search"},
        {"Stage": "Fusion", "Mile-1": "Reciprocal rank fusion", "This project": "Relative-score weighted fusion; RRF also available"},
        {"Stage": "Reranking", "Mile-1": "BGE cross-encoder", "This project": "MMR diversity or relevance ordering"},
        {"Stage": "Context", "Mile-1": "Parent lookup", "This project": "Query-focused sentence extraction or neighbor windows"},
        {"Stage": "Generation", "Mile-1": "Groq + citation prompt", "This project": "Groq + citation-ID checks + explicit extractive fallback"},
    ]), hide_index=True, width="stretch")
    st.markdown("**What the methods actually do**")
    st.write("PRF adds terms from three initial TF-IDF matches. Rocchio moves the query vector toward their centroid. Routing softly boosts a domain rather than discarding other evidence. MMR trades relevance against similarity to already selected chunks. Context compression selects original sentences; it does not generate evidence.")
    st.write("Multi-query and HyDE call Groq when configured. HyDE text is used only for retrieval. It is never passed off as a source. LSA is a corpus-trained linear latent space, not a multilingual pretrained encoder. Arabic support without Groq is a limited finance glossary, not general translation.")
    st.caption("Citation validation checks IDs and paragraph coverage, not whether every generated claim is entailed. No external database, Elasticsearch server, FastAPI service, GPU or model download is required.")
    st.markdown("**Project owner:** [John Khamis](https://github.com/Yassahany2004)")
