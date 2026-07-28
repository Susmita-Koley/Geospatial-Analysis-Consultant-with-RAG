"""
app.py — Streamlit UI for Project 1: Geospatial Document Intelligence

Tabs:
  🔍 Query        — Ask questions, get cited answers from the RAG pipeline
  📊 Evaluation   — RAGAS metric dashboard (radar chart + per-question table)
  ⚙️  Setup        — One-click ingestion + status indicators
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="GeoDoc Intelligence | RAG System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)  # suppress verbose logs in UI


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark space aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Gradient hero */
.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 50%, #0F172A 100%);
    border: 1px solid #1E40AF44;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 32px rgba(56, 189, 248, 0.08);
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38BDF8, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub {
    color: #94A3B8;
    font-size: 0.95rem;
    margin-top: 0.4rem;
}

/* Citation cards */
.citation-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-left: 3px solid #38BDF8;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.85rem;
    color: #CBD5E1;
}

/* Metric pill */
.metric-pill {
    display: inline-block;
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.8rem;
    color: #38BDF8;
    margin: 0.2rem;
}

/* Answer box */
.answer-box {
    background: #0F172A;
    border: 1px solid #1E40AF55;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    line-height: 1.7;
}

/* Status indicators */
.status-ok   { color: #4ADE80; font-weight: 600; }
.status-warn { color: #FBBF24; font-weight: 600; }
.status-err  { color: #F87171; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-banner">
  <p class="hero-title">🛰️ Geospatial Document Intelligence</p>
  <p class="hero-sub">
    RAG-powered Q&amp;A over official ESA, NASA &amp; USGS geospatial documentation
    — with source citations and RAGAS quality evaluation
  </p>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — document corpus info
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Corpus")
    docs_dir = ROOT / "docs"
    pdf_files = sorted(docs_dir.glob("*.pdf")) if docs_dir.exists() else []
    if pdf_files:
        for pdf in pdf_files:
            size_kb = pdf.stat().st_size // 1024
            st.markdown(f"- `{pdf.name}` ({size_kb} KB)")
    else:
        st.warning("No PDFs found in `docs/`")

    st.divider()
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Retrieved chunks (top-k)", min_value=2, max_value=10, value=5)

    st.divider()
    st.markdown(
        """
    **Tech Stack**
    - LangChain · ChromaDB
    - Groq API (llama3-8b)
    - SentenceTransformers
    - RAGAS Evaluation
    - PyMuPDF
    """
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_query, tab_eval, tab_setup = st.tabs(["🔍 Query", "📊 Evaluation", "⚙️ Setup"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Query Interface
# ══════════════════════════════════════════════════════════════════════════════
with tab_query:
    st.markdown("#### Ask anything about geospatial sensors & products")

    # Example queries
    st.markdown("**Example questions:**")
    example_qs = [
        "What are the spectral bands of Sentinel-2 MSI and their spatial resolutions?",
        "How does MODIS LST product MOD11A1 correct for atmospheric effects?",
        "What is the radiometric resolution of Landsat Collection 2 Level-1 products?",
        "Explain the difference between NDVI and EVI in the MODIS vegetation index product.",
        "What preprocessing steps are applied to Sentinel-2 Level-2A products?",
    ]
    cols = st.columns(2)
    selected_q = None
    for i, eq in enumerate(example_qs):
        if cols[i % 2].button(f"💬 {eq[:55]}…", key=f"ex_{i}", use_container_width=True):
            selected_q = eq

    st.divider()

    question = st.text_area(
        "Your question",
        value=selected_q or "",
        height=80,
        placeholder="e.g. What spectral bands does Sentinel-2 have?",
        key="question_input",
    )

    ask_col, clear_col = st.columns([4, 1])
    ask_btn = ask_col.button("🚀 Ask", type="primary", use_container_width=True)
    if clear_col.button("🗑️ Clear", use_container_width=True):
        st.rerun()

    if ask_btn and question.strip():
        try:
            from src.rag.retriever import GeospatialRAG

            with st.spinner("Retrieving relevant document chunks and generating answer…"):
                rag = st.session_state.get("rag_instance")
                if rag is None:
                    rag = GeospatialRAG()
                    st.session_state["rag_instance"] = rag

                response = rag.query(question.strip(), top_k=top_k)

            # ── Answer ────────────────────────────────────────────────────────
            st.markdown("#### 💡 Answer")
            st.markdown(
                f'<div class="answer-box">{response.answer}</div>',
                unsafe_allow_html=True,
            )

            # ── Citations ─────────────────────────────────────────────────────
            st.markdown("#### 📎 Source Citations")
            if response.citations:
                for c in response.citations:
                    st.markdown(
                        f'<div class="citation-card">📄 <strong>{c.source}</strong> — Page {c.page}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No citations extracted.")

            # ── Retrieved chunks (expandable) ─────────────────────────────────
            with st.expander(f"🔎 View {len(response.context_chunks)} retrieved context chunks"):
                for i, chunk in enumerate(response.context_chunks):
                    st.markdown(f"**Chunk {i+1}**")
                    st.text(chunk[:600] + ("…" if len(chunk) > 600 else ""))
                    st.divider()

        except RuntimeError as e:
            st.error(str(e))
            st.info("👉 Go to the **⚙️ Setup** tab and click **Build Vector Store** first.")
        except EnvironmentError as e:
            st.error(str(e))

    elif ask_btn:
        st.warning("Please enter a question.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAGAS Evaluation Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("#### 📊 RAGAS Evaluation Dashboard")
    st.markdown(
        "RAGAS measures four orthogonal dimensions of RAG quality — "
        "each one answers a different failure mode."
    )

    metric_info = {
        "faithfulness": (
            "Faithfulness",
            "Are claims in the answer grounded in the retrieved context?",
            "#38BDF8",
        ),
        "answer_relevancy": (
            "Answer Relevancy",
            "Does the answer actually address the question?",
            "#818CF8",
        ),
        "context_precision": (
            "Context Precision",
            "Are the retrieved chunks relevant (signal-to-noise)?",
            "#4ADE80",
        ),
        "context_recall": (
            "Context Recall",
            "Does the context contain what's needed to answer?",
            "#FBBF24",
        ),
    }

    eval_results_path = ROOT / "data" / "eval_results.csv"

    if eval_results_path.exists():
        import pandas as pd
        import plotly.graph_objects as go

        df = pd.read_csv(eval_results_path)
        metric_cols = [c for c in metric_info.keys() if c in df.columns]
        avg = df[metric_cols].mean()

        # ── Score cards ───────────────────────────────────────────────────────
        score_cols = st.columns(len(metric_cols))
        for i, col_name in enumerate(metric_cols):
            label, desc, color = metric_info[col_name]
            val = avg[col_name]
            score_cols[i].metric(label=label, value=f"{val:.3f}", help=desc)

        st.divider()

        # ── Radar chart ───────────────────────────────────────────────────────
        fig = go.Figure()
        categories = [metric_info[m][0] for m in metric_cols]
        values = [avg[m] for m in metric_cols]
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.15)",
                line=dict(color="#38BDF8", width=2),
                name="RAG System",
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=12)),
                bgcolor="#1E293B",
            ),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font=dict(color="#E2E8F0"),
            showlegend=False,
            height=400,
            margin=dict(l=60, r=60, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Per-question table ────────────────────────────────────────────────
        with st.expander("📋 Per-question results"):
            display_cols = (
                ["question"] if "question" in df.columns else ["user_input"]
            ) + metric_cols
            st.dataframe(
                df[display_cols].style.background_gradient(
                    subset=metric_cols, cmap="Blues", vmin=0, vmax=1
                ),
                use_container_width=True,
            )

    else:
        st.info(
            "No evaluation results found yet.\n\n"
            "**Steps to generate:**\n"
            "1. Build the vector store (⚙️ Setup tab)\n"
            "2. Run `python scripts/generate_testset.py` in your terminal\n"
            "3. Run `python -m src.rag.evaluator` in your terminal\n"
            "4. Refresh this tab — results will appear here."
        )

        # Show metric explanations as preview
        st.markdown("#### What each RAGAS metric measures")
        for key, (label, desc, color) in metric_info.items():
            st.markdown(f'<span class="metric-pill">◉ {label}</span>', unsafe_allow_html=True)
            st.caption(desc)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Setup / Ingestion
# ══════════════════════════════════════════════════════════════════════════════
with tab_setup:
    st.markdown("#### ⚙️ Setup & Ingestion Status")

    chroma_dir = ROOT / "data" / "chroma_db"
    testset_path = ROOT / "data" / "testset.json"

    # Status checks
    col1, col2, col3 = st.columns(3)
    with col1:
        pdf_count = len(list((ROOT / "docs").glob("*.pdf"))) if (ROOT / "docs").exists() else 0
        status = "✅ OK" if pdf_count == 7 else ("⚠️ Partial" if pdf_count > 0 else "❌ Missing")
        st.metric("PDF Documents", f"{pdf_count}/7", delta=status)

    with col2:
        chroma_ok = chroma_dir.exists() and any(chroma_dir.iterdir()) if chroma_dir.exists() else False
        st.metric("Vector Store", "✅ Ready" if chroma_ok else "❌ Not built")

    with col3:
        testset_ok = testset_path.exists()
        st.metric("Testset", "✅ Generated" if testset_ok else "❌ Not generated")

    st.divider()

    # ── Build vector store ────────────────────────────────────────────────────
    st.markdown("##### Step 1 — Build Vector Store")
    st.caption(
        "Parses all 7 PDFs, splits into chunks, embeds with SentenceTransformers "
        "(all-MiniLM-L6-v2, runs locally), and persists to ChromaDB. "
        "Takes ~2–4 minutes on first run."
    )

    rebuild = st.checkbox("Force rebuild (re-embed everything)", value=False)
    if st.button("🏗️ Build Vector Store", type="primary", key="build_btn"):
        try:
            from src.rag.ingest import build_vectorstore

            with st.spinner("Building vector store… this takes 2–4 minutes on first run."):
                vs = build_vectorstore(force_rebuild=rebuild)
                collection = vs._collection
                count = collection.count()
            st.success(f"✅ Vector store built! {count:,} chunks indexed in ChromaDB.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error during ingestion: {e}")

    st.divider()

    # ── Generate testset ──────────────────────────────────────────────────────
    st.markdown("##### Step 2 — Generate Evaluation Testset")
    st.caption(
        "Uses RAGAS `TestsetGenerator` to auto-create 60 QA pairs from your documents. "
        "Run this from the terminal (not here) to avoid Streamlit timeouts:"
    )
    st.code("python scripts/generate_testset.py", language="bash")

    st.divider()

    # ── Run evaluation ────────────────────────────────────────────────────────
    st.markdown("##### Step 3 — Run RAGAS Evaluation")
    st.caption(
        "Scores the RAG system on all 60 QA pairs. Saves results to "
        "`data/eval_results.csv` for the dashboard."
    )
    st.code("python -m src.rag.evaluator", language="bash")

    st.divider()

    # ── Environment info ──────────────────────────────────────────────────────
    st.markdown("##### Environment Info")
    import platform
    env_data = {
        "Python": platform.python_version(),
        "OS": platform.system() + " " + platform.release(),
        "Chroma DB path": str(chroma_dir),
        "Docs path": str(ROOT / "docs"),
    }
    for k, v in env_data.items():
        st.markdown(f"- **{k}**: `{v}`")
