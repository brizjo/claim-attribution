"""
RAG Claim Attribution System — Interfaccia Streamlit.

Applicazione principale che orchestra la pipeline completa:
1. Input query utente
2. Retrieval da Wikipedia (ChromaDB)
3. Generazione con Llama-3 (Ollama)
4. Segmentazione sentence-level
5. Calcolo matrice di supporto (BERTScore + ROUGE-L)
6. Visualizzazione Highlight Gradient

Avvio: streamlit run app.py
"""

import sys
from pathlib import Path

# Aggiungi la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import numpy as np

from config import settings
from src.retriever.wiki_retriever import WikiRetriever
from src.generator.llama_generator import LlamaGenerator
from src.segmentation.sentence_splitter import SentenceSplitter
from src.attribution.matrix import SupportMatrix
from src.visualization.highlight_renderer import HighlightRenderer


# ====================================================================
# Page Configuration
# ====================================================================

st.set_page_config(
    page_title="RAG Claim Attribution",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================================
# Custom CSS — Dark Premium Theme
# ====================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 300;
    }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .status-online {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Cards */
    .info-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(8px);
    }
    .info-card h4 {
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .info-card p {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Context documents */
    .context-doc {
        background: rgba(30, 41, 59, 0.4);
        border-left: 3px solid #6366f1;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #cbd5e1;
    }
    .context-source {
        color: #818cf8;
        font-size: 0.78rem;
        font-weight: 500;
        margin-top: 0.4rem;
    }

    /* Metrics row */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-box {
        flex: 1;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Pipeline step indicators */
    .pipeline-steps {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
    }
    .pipeline-step.active {
        background: rgba(99, 102, 241, 0.25);
        border-color: rgba(99, 102, 241, 0.5);
        color: #c7d2fe;
    }

    /* Divider */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148,163,184,0.2), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ====================================================================
# Cached Initialization
# ====================================================================

@st.cache_resource
def init_retriever():
    """Inizializza il retriever (cached per sessione)."""
    return WikiRetriever()


@st.cache_resource
def init_generator():
    """Inizializza il generator (cached per sessione)."""
    return LlamaGenerator()


@st.cache_resource
def init_splitter():
    """Inizializza il sentence splitter (cached per sessione)."""
    return SentenceSplitter()


@st.cache_resource
def init_renderer():
    """Inizializza il renderer (cached per sessione)."""
    return HighlightRenderer()


# ====================================================================
# Header
# ====================================================================

st.markdown("""
<div class="main-header">
    <h1>🔍 RAG Claim Attribution</h1>
    <p>Post-Retrieval attribution with highlight gradient visualization</p>
</div>
""", unsafe_allow_html=True)

# Pipeline steps
st.markdown("""
<div class="pipeline-steps">
    <span class="pipeline-step">📥 Retrieval</span>
    <span class="pipeline-step">→</span>
    <span class="pipeline-step">🤖 Generation</span>
    <span class="pipeline-step">→</span>
    <span class="pipeline-step">✂️ Segmentation</span>
    <span class="pipeline-step">→</span>
    <span class="pipeline-step">📊 Attribution Matrix</span>
    <span class="pipeline-step">→</span>
    <span class="pipeline-step">🎨 Highlight</span>
</div>
<hr class="section-divider">
""", unsafe_allow_html=True)

# ====================================================================
# Sidebar — Configuration & Status
# ====================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Model status
    generator = init_generator()
    ollama_status = generator.is_available()
    if ollama_status:
        st.markdown(
            '<span class="status-badge status-online">● Ollama Online</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-offline">● Ollama Offline</span>',
            unsafe_allow_html=True,
        )
        st.warning("Avvia Ollama con: `ollama serve`")

    st.markdown("---")

    # Parametri configurabili
    st.markdown("#### 📐 Retrieval")
    top_k = st.slider("Top-K documents", 1, 10, settings.TOP_K_DOCUMENTS)

    st.markdown("#### 🧮 Attribution Weights")
    w_bert = st.slider(
        "BERTScore weight",
        0.0, 1.0, settings.WEIGHT_BERTSCORE, 0.05,
    )
    w_lex = st.slider(
        "Lexical weight",
        0.0, 1.0, settings.WEIGHT_LEXICAL_OVERLAP, 0.05,
    )

    st.markdown("#### 🎨 Thresholds")
    th_high = st.slider(
        "High support ≥", 0.0, 1.0, settings.SUPPORT_THRESHOLD_HIGH, 0.05
    )
    th_med = st.slider(
        "Medium support ≥", 0.0, 1.0, settings.SUPPORT_THRESHOLD_MEDIUM, 0.05
    )
    th_low = st.slider(
        "Low support ≥", 0.0, 1.0, settings.SUPPORT_THRESHOLD_LOW, 0.05
    )

    st.markdown("---")

    # Index building
    st.markdown("#### 📚 Knowledge Base")
    st.markdown(f"<span style='font-size:0.8rem;color:#94a3b8'>Dataset: {settings.HF_DATASET_NAME}</span>", unsafe_allow_html=True)
    if st.button("🔄 Build Dataset Index", use_container_width=True):
        with st.spinner("Downloading and indexing dataset documents..."):
            retriever = init_retriever()
            n_chunks = retriever.build_index()
            st.success(f"✅ Indexed {n_chunks} chunks")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:0.75rem;'>"
        "RAG Claim Attribution v0.1<br>"
        "University Research Project"
        "</div>",
        unsafe_allow_html=True,
    )

# ====================================================================
# Main Content — Query & Pipeline
# ====================================================================

# Query input
query = st.text_input(
    "🔎 Enter your question",
    placeholder="e.g., What is retrieval-augmented generation?",
    help="The system will retrieve relevant Wikipedia documents and generate an attributed answer.",
)

# Run pipeline
if query:
    retriever = init_retriever()
    splitter = init_splitter()
    renderer = init_renderer()

    # Override renderer thresholds from sidebar
    renderer.threshold_high = th_high
    renderer.threshold_medium = th_med
    renderer.threshold_low = th_low

    # ── Step 1: Retrieval ──────────────────────────────────────
    with st.status("📥 Retrieving documents from Wikipedia...", expanded=False) as status:
        try:
            retrieved_docs = retriever.query(query, top_k=top_k)
            if not retrieved_docs:
                st.warning(
                    "⚠️ No documents found. Build the index first using the sidebar button."
                )
                st.stop()
            status.update(
                label=f"📥 Retrieved {len(retrieved_docs)} documents",
                state="complete",
            )
        except Exception as e:
            status.update(label=f"❌ Retrieval failed: {e}", state="error")
            st.stop()

    # ── Step 2: Generation ─────────────────────────────────────
    with st.status("🤖 Generating response with Llama-3...", expanded=False) as status:
        try:
            result = generator.run(query, retrieved_docs)
            generated_response = result["response"]
            status.update(label="🤖 Response generated", state="complete")
        except ConnectionError:
            st.error(
                "❌ Cannot connect to Ollama. Make sure it's running:\n\n"
                "```bash\nollama serve\n```"
            )
            st.stop()
        except Exception as e:
            status.update(label=f"❌ Generation failed: {e}", state="error")
            st.stop()

    # ── Step 3: Segmentation ───────────────────────────────────
    with st.status("✂️ Segmenting into sentences...", expanded=False) as status:
        # Segmenta i documenti di contesto in frasi singole
        evidence_sentences = splitter.split_documents(retrieved_docs)

        # Estrai fatti atomici dalla risposta generata
        atomic_facts = splitter.extract_atomic_facts(generated_response)

        status.update(
            label=f"✂️ {len(evidence_sentences)} evidence sentences, "
                  f"{len(atomic_facts)} atomic facts",
            state="complete",
        )

    # ── Step 4: Attribution Matrix ─────────────────────────────
    with st.status("📊 Computing support matrix...", expanded=False) as status:
        support_matrix = SupportMatrix(
            weight_bertscore=w_bert,
            weight_lexical=w_lex,
        )
        matrix = support_matrix.compute(atomic_facts, evidence_sentences)
        attribution_results = support_matrix.get_attribution_results()
        overall_score = support_matrix.get_overall_score()

        status.update(
            label=f"📊 Attribution complete — Overall: {overall_score*100:.0f}%",
            state="complete",
        )

    # ── Step 5: Visualization ──────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Metrics overview
    n_high = sum(1 for r in attribution_results if r.support_score >= th_high)
    n_halluc = sum(1 for r in attribution_results if r.support_score < th_low)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score_color = renderer._score_to_border(overall_score)
        st.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-label'>Overall Score</div>"
            f"<div class='metric-value' style='color:{score_color}'>"
            f"{overall_score*100:.0f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-label'>Atomic Facts</div>"
            f"<div class='metric-value' style='color:#a5b4fc'>"
            f"{len(atomic_facts)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-label'>Well Supported</div>"
            f"<div class='metric-value' style='color:#10b981'>"
            f"{n_high}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-label'>Hallucination Risk</div>"
            f"<div class='metric-value' style='color:#ef4444'>"
            f"{n_halluc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Highlight Gradient Response ────────────────────────────
    st.markdown("### 🎨 Attributed Response")
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;'>"
        "Hover over each segment to see attribution details. "
        "Colors indicate the level of source support.</p>",
        unsafe_allow_html=True,
    )

    highlighted_html = renderer.render_response(
        attribution_results, overall_score
    )
    st.markdown(highlighted_html, unsafe_allow_html=True)

    # ── Expandable sections ────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Context documents
    with st.expander("📄 Retrieved Context Documents", expanded=False):
        for i, doc in enumerate(retrieved_docs):
            source = doc.get("metadata", {}).get("source", "Unknown")
            distance = doc.get("distance", 0)
            st.markdown(
                f"<div class='context-doc'>"
                f"{doc['document']}"
                f"<div class='context-source'>📌 {source} — "
                f"Distance: {distance:.4f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Attribution details table
    with st.expander("📋 Attribution Details", expanded=False):
        for i, result in enumerate(attribution_results):
            score_color = renderer._score_to_border(result.support_score)
            st.markdown(
                f"**Fact {i+1}:** {result.fact}\n\n"
                f"- **Support Score:** "
                f"<span style='color:{score_color}'>"
                f"{result.support_score*100:.1f}%</span>\n"
                f"- **BERTScore F1:** {result.bertscore:.3f}\n"
                f"- **Lexical (ROUGE-L):** {result.lexical_score:.3f}\n"
                f"- **Best Evidence:** _{result.best_evidence[:200]}_\n"
                f"- **Source:** {result.best_evidence_source}\n\n---",
                unsafe_allow_html=True,
            )

    # Support Matrix heatmap
    with st.expander("🧮 Support Matrix Heatmap", expanded=False):
        if matrix is not None and matrix.size > 0:
            evidence_labels = [ev["sentence"] for ev in evidence_sentences]
            heatmap_html = renderer.render_matrix_heatmap(
                matrix, atomic_facts, evidence_labels
            )
            st.markdown(heatmap_html, unsafe_allow_html=True)
        else:
            st.info("No matrix data available.")

else:
    # Landing state — show instructions
    st.markdown("""
    <div class="info-card">
        <h4>👋 Welcome to RAG Claim Attribution</h4>
        <p>
            This system implements <strong>Post-Retrieval claim attribution</strong>
            with a visual <strong>highlight gradient</strong> to show how well each
            part of the generated response is supported by the source documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🚀 Getting Started</h4>
            <p>
                1. Make sure <strong>Ollama</strong> is running with Llama-3<br>
                2. Click <strong>"Build Dataset Index"</strong> in the sidebar<br>
                3. Type your question in the search bar above<br>
                4. Explore the attributed response with color coding
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🎨 Color Legend</h4>
            <p>
                🟢 <strong>Green</strong> — High support (≥80%)<br>
                🟡 <strong>Yellow</strong> — Partial support (50-80%)<br>
                🟠 <strong>Orange</strong> — Weak support (30-50%)<br>
                🔴 <strong>Red</strong> — Hallucination risk (&lt;30%)
            </p>
        </div>
        """, unsafe_allow_html=True)
