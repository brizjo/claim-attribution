"""
Claim Attribution — LPG/Neo4j Streamlit App.

Tab 1 — Ingest: upload PDF/TXT → clean (Llama-3) → extract triples (REBEL) → Neo4j
Tab 2 — Claim Attribution: input claim → exact match / semantic fallback → source

Run: streamlit run app.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ["HF_HOME"] = r"D:\hf_home"
os.environ["HF_HUB_CACHE"] = r"D:\hf_home\hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\hf_home\hub"
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_home\transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"D:\hf_home\sentence_transformers"
os.environ["HF_DATASETS_CACHE"] = r"D:\hf_home\datasets"
os.environ["TORCH_HOME"] = r"D:\hf_home\torch"

import streamlit as st

from config import settings

st.set_page_config(
    page_title="Claim Attribution",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.main-header { text-align:center; padding:1.5rem 0 0.5rem; }
.main-header h1 {
    font-size:2.2rem; font-weight:700;
    background:linear-gradient(135deg,#6366f1,#a78bfa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.badge-online  { display:inline-block; padding:3px 12px; border-radius:20px;
    background:rgba(16,185,129,.15); color:#10b981;
    border:1px solid rgba(16,185,129,.3); font-size:.8rem; font-weight:500; }
.badge-offline { display:inline-block; padding:3px 12px; border-radius:20px;
    background:rgba(239,68,68,.15); color:#ef4444;
    border:1px solid rgba(239,68,68,.3); font-size:.8rem; font-weight:500; }
.card {
    background:rgba(30,41,59,.5); border:1px solid rgba(148,163,184,.1);
    border-radius:12px; padding:1.2rem; margin:.5rem 0; }
.card h4 { color:#e2e8f0; margin-bottom:.5rem; font-weight:600; }
.card p  { color:#94a3b8; font-size:.9rem; line-height:1.6; }
.triple-tag {
    display:inline-block; background:rgba(99,102,241,.15);
    border:1px solid rgba(99,102,241,.3); border-radius:8px;
    padding:2px 10px; color:#a5b4fc; font-size:.82rem; margin:2px; }
.triple-tag-answer {
    display:inline-block; background:rgba(16,185,129,.2);
    border:1px solid rgba(16,185,129,.5); border-radius:8px;
    padding:2px 10px; color:#34d399; font-size:.82rem; margin:2px;
    font-weight:600; box-shadow:0 0 8px rgba(16,185,129,.3); }
.result-exact    { background:rgba(16,185,129,.1);  border-left:4px solid #10b981; padding:1rem; border-radius:0 10px 10px 0; }
.result-semantic { background:rgba(245,158,11,.1);  border-left:4px solid #f59e0b; padding:1rem; border-radius:0 10px 10px 0; }
.result-notfound { background:rgba(239,68,68,.1);   border-left:4px solid #ef4444; padding:1rem; border-radius:0 10px 10px 0; }
.chunk-box {
    background:rgba(15,23,42,.6); border:1px solid rgba(99,102,241,.2);
    border-radius:8px; padding:1rem; font-size:.9rem; line-height:1.7;
    color:#cbd5e1; font-style:italic; margin-top:.8rem; }
</style>
""", unsafe_allow_html=True)


# ====================================================================
# Cached resources
# ====================================================================

@st.cache_resource
def get_neo4j_client():
    try:
        from src.graph.neo4j_client import Neo4jClient
        return Neo4jClient()
    except Exception as e:
        return None


@st.cache_resource
def get_generator(model_name: str = settings.OLLAMA_MODEL):
    from src.generator.llama_generator import LlamaGenerator
    return LlamaGenerator(model=model_name)


# ====================================================================
# Header
# ====================================================================

st.markdown("""
<div class="main-header">
    <h1>🧠 Claim Attribution — LPG/Neo4j</h1>
</div>
""", unsafe_allow_html=True)


# ====================================================================
# Sidebar — status
# ====================================================================

with st.sidebar:
    st.markdown("### System Status")

    neo4j = get_neo4j_client()
    if neo4j and neo4j.is_connected():
        st.markdown('<span class="badge-online">● Neo4j Connected</span>', unsafe_allow_html=True)
        stats = neo4j.stats()
        st.caption(f"Entities: **{stats['nodes']}** | Relations: **{stats['relations']}**")
        active_db = getattr(neo4j, "database", None)
        if active_db:
            st.caption(f"DB attiva: `{active_db}` — assicurati che Browser punti qui")
        list_dbs = getattr(neo4j, "list_databases", None)
        if callable(list_dbs):
            dbs = list_dbs()
            if dbs:
                st.caption(f"DB visibili: {', '.join(dbs)}")
        if st.button("♻️ Reload Neo4j Client"):
            get_neo4j_client.clear()
            st.rerun()
    else:
        st.markdown('<span class="badge-offline">● Neo4j Offline</span>', unsafe_allow_html=True)
        st.warning(
            "Start Neo4j Desktop, open a database, then set:\n"
            "```\nNEO4J_PASSWORD=yourpass\n```\nin your environment."
        )
        if st.button("🔄 Retry Connection"):
            get_neo4j_client.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("#### Processed Documents")
    if neo4j and neo4j.is_connected():
        docs = neo4j.get_documents()
        if docs:
            import pandas as pd
            df = pd.DataFrame(docs)[["name", "ingested_at", "num_chunks", "num_triples", "status"]]
            df.columns = ["File", "Ingested At", "Chunks", "Triples", "Status"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No documents ingested yet.")
    else:
        st.caption("Neo4j not connected.")

    st.markdown("---")

    st.markdown("#### Configurazione Modello LLM")
    selected_model = st.selectbox(
        "Modello di riscrittura",
        options=["qwen2.5:1.5b", "llama3.2", "llama3"],
        index=0,
    )
    
    if selected_model == "qwen2.5:1.5b":
        st.info("⚡ **Tempo stimato**: ~1s per chunk (Molto Veloce)")
    elif selected_model == "llama3.2":
        st.info("🚀 **Tempo stimato**: ~3s per chunk (Veloce)")
    else:
        st.warning("🐢 **Tempo stimato**: ~10s per chunk (Lento, sconsigliato)")

    gen = get_generator(selected_model)
    if gen.is_available():
        st.markdown('<span class="badge-online">● Ollama Online</span>', unsafe_allow_html=True)
        st.caption(f"Model in uso: **{gen.model}**")
        if st.button("🔄 Reload LLM Cache"):
            get_generator.clear()
            st.rerun()
    else:
        st.markdown('<span class="badge-offline">● Ollama Offline</span>', unsafe_allow_html=True)
        st.caption(f"Il modello {selected_model} non è stato trovato. Assicurati di averne completato il download.")

    st.markdown("---")
    st.markdown("#### Settings")
    semantic_threshold = st.slider(
        "Semantic similarity threshold",
        0.5, 1.0, settings.SEMANTIC_THRESHOLD, 0.05,
    )
    st.markdown("---")
    st.caption("Claim Attribution v1.0 — LPG/Neo4j")


# ====================================================================
# Tabs
# ====================================================================

tab_ingest, tab_claim = st.tabs(["📂 Ingest Documents", "🔍 Claim Attribution"])


# ──────────────────────────────────────────────────────────────────────
# TAB 1 — INGEST
# ──────────────────────────────────────────────────────────────────────

with tab_ingest:
    st.markdown("### Upload Documents")
    st.markdown(
        "<p style='color:#94a3b8;font-size:.9rem;'>"
        "Supports PDF and TXT. Each file is chunked → coreference resolved (optional, spaCy+coreferee) "
        "→ triple extracted (REBEL) → indexed in Neo4j."
        "</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    st.caption(
        "🧹 Coreference resolution: **always on** — required for high-quality "
        "triple extraction (pronouns/aliases resolved before mREBEL)."
    )

    run_clustering = st.checkbox(
        "🔗 Run Entity Clustering after indexing",
        value=False,
        help="Merges semantically-equivalent entity nodes (cosine ≥ 0.90). Slower on large graphs."
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        process_btn = st.button(
            "⚙️ Process & Index",
            type="primary",
            use_container_width=True,
            disabled=(not uploaded or neo4j is None),
        )
    with col_btn2:
        clear_btn = st.button(
            "🗑️ Clear Graph",
            use_container_width=True,
            disabled=(neo4j is None),
        )

    if clear_btn and neo4j:
        neo4j.clear_graph()
        st.cache_resource.clear()
        st.success("Graph cleared.")
        st.rerun()

    if process_btn and uploaded and neo4j:
        import time
        from src.ingestion.document_loader import DocumentLoader
        from src.ingestion.coref_resolver import CoreferenceResolver
        from src.ingestion.triple_extractor import TripleExtractor
        from src.ingestion.graph_writer import GraphWriter

        loader = DocumentLoader()
        resolver = CoreferenceResolver()
        extractor = TripleExtractor()
        writer = GraphWriter(client=neo4j)

        total_triples = 0
        total_chunks = 0
        timings = {"load": 0, "coref": 0, "extract": 0, "embed": 0, "index": 0}

        for ufile in uploaded:
            tmp_path = Path(f"D:/hf_home/tmp_{ufile.name}")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(ufile.read())

            st.markdown(f"#### Processing: `{ufile.name}`")

            # ── Load ──
            with st.status(f"📄 Loading `{ufile.name}`...", expanded=True) as stage:
                t0 = time.time()
                chunks = loader.load(tmp_path)
                total_chunks += len(chunks)
                timings["load"] += time.time() - t0
                stage.update(label=f"📄 Loaded {len(chunks)} chunks ({timings['load']:.1f}s)", state="complete")

            # ── Minibatch streaming: groups of REBEL_BATCH_SIZE chunks ──
            #    Each batch: coref each chunk → batched mREBEL forward →
            #    restore original chunk_text → embed + write to Neo4j.
            #    Crash mid-file = batches already written persist; document
            #    marked "error" on finalize.
            file_triples = 0
            file_status = "done"
            batch_size = settings.REBEL_BATCH_SIZE
            n_batches = (len(chunks) + batch_size - 1) // batch_size

            with st.status(
                f"🌀 Streaming {len(chunks)} chunks in {n_batches} minibatches "
                f"of {batch_size} → Neo4j...",
                expanded=True,
            ) as stage:
                t_stream = time.time()

                for bi in range(n_batches):
                    batch = chunks[bi * batch_size:(bi + 1) * batch_size]
                    originals = [c["text"] for c in batch]

                    # 1. Coref each chunk in batch (mandatory)
                    t0 = time.time()
                    resolved_batch = []
                    for chunk, orig in zip(batch, originals):
                        resolved_batch.append({**chunk, "text": resolver.resolve(orig)})
                    timings["coref"] += time.time() - t0

                    # 2. Batched mREBEL forward — internal batching uses
                    #    REBEL_BATCH_SIZE so 1 forward pass per minibatch.
                    t0 = time.time()
                    try:
                        triples = extractor.extract(resolved_batch)
                    except Exception as e:
                        stage.update(label=f"⚠️ Batch {bi+1}/{n_batches}: extract failed ({e})")
                        file_status = "error"
                        continue
                    timings["extract"] += time.time() - t0

                    # 3. Restore ORIGINAL chunk_text per triple (lookup by
                    #    chunk_index) — evidence must be source verbatim.
                    orig_by_idx = {c["chunk_index"]: c["text"] for c in batch}
                    triples = [
                        t._replace(chunk_text=orig_by_idx.get(t.chunk_index, t.chunk_text))
                        for t in triples
                    ]

                    # 4. Embed + write batch
                    t0 = time.time()
                    try:
                        written = writer.write_triples(triples)
                    except Exception as e:
                        stage.update(label=f"⚠️ Batch {bi+1}/{n_batches}: write failed ({e})")
                        file_status = "error"
                        continue
                    elapsed = time.time() - t0
                    timings["embed"] += elapsed * 0.7
                    timings["index"] += elapsed * 0.3

                    file_triples += written
                    total_triples += written

                    chunks_done = min((bi + 1) * batch_size, len(chunks))
                    stage.update(
                        label=(
                            f"🌀 Batch {bi+1}/{n_batches} ({chunks_done}/"
                            f"{len(chunks)} chunks) → +{written} triples "
                            f"(file total: {file_triples}, "
                            f"{time.time() - t_stream:.1f}s)"
                        )
                    )

                # 5. Finalize Document node ONCE per file
                writer.finalize_document(
                    source_file=ufile.name,
                    num_chunks=len(chunks),
                    num_triples=file_triples,
                    status=file_status,
                )
                stage.update(
                    label=(
                        f"✅ {ufile.name}: {file_triples} triples across "
                        f"{len(chunks)} chunks in {n_batches} batches "
                        f"({time.time() - t_stream:.1f}s)"
                    ),
                    state="complete",
                )

            tmp_path.unlink(missing_ok=True)

        # Entity clustering (optional)
        if run_clustering:
            from src.ingestion.entity_clusterer import EntityClusterer
            with st.status("🔗 Clustering entities...", expanded=False) as stage:
                clusterer = EntityClusterer(client=neo4j)
                merges = clusterer.cluster()
                stage.update(
                    label=f"🔗 Entity clustering complete — {merges} merge(s) performed",
                    state="complete"
                )

        # Show timing breakdown
        st.markdown("### Performance Breakdown")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Load", f"{timings['load']:.1f}s")
        with col2:
            st.metric("Coref", f"{timings['coref']:.1f}s")
        with col3:
            st.metric("Extract", f"{timings['extract']:.1f}s")
        with col4:
            st.metric("Embed", f"{timings['embed']:.1f}s")
        with col5:
            st.metric("Index", f"{timings['index']:.1f}s")

        st.success(
            f"Done! {len(uploaded)} file(s) → {total_chunks} chunks → "
            f"**{total_triples} triples** indexed in Neo4j."
        )
        st.cache_resource.clear()
        st.rerun()

    if not uploaded:
        st.markdown("""
        <div class="card">
            <h4>Pipeline</h4>
            <p>
                1. <strong>Load</strong> — PDF/TXT → word chunks (200 words, 50 overlap)<br>
                2. <strong>Clean</strong> — spaCy+coreferee resolves coreferences &amp; normalizes entities<br>
                3. <strong>Extract</strong> — REBEL extracts (Subject, Predicate, Object) triples<br>
                4. <strong>Index</strong> — Triples written to Neo4j with predicate embeddings
            </p>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# TAB 2 — CLAIM ATTRIBUTION
# ──────────────────────────────────────────────────────────────────────

with tab_claim:
    st.markdown("### Verifica Claim o Domanda")
    st.markdown(
        "<p style='color:#94a3b8;font-size:.9rem;'>"
        "Inserisci un'affermazione (claim) <em>oppure</em> una domanda. "
        "Le affermazioni vengono parsate via mREBEL e verificate sul grafo. "
        "Le domande vengono convertite in tripla parziale via LLM e risolte "
        "tramite pattern query + cosine similarity sul predicato."
        "</p>",
        unsafe_allow_html=True,
    )

    claim_input = st.text_area(
        "Claim o domanda",
        placeholder="es. 'Tenma è il protagonista di Monster' oppure 'Chi è il protagonista di Monster?'",
        height=80,
    )

    verify_btn = st.button(
        "🔍 Verifica / Rispondi",
        type="primary",
        disabled=(not claim_input.strip() or neo4j is None),
    )

    if neo4j is None:
        st.warning("Neo4j not connected. Start Neo4j Desktop first.")

    if verify_btn and claim_input.strip() and neo4j:
        from src.attribution.claim_attributor import ClaimAttributor

        attributor = ClaimAttributor(
            client=neo4j,
            semantic_threshold=semantic_threshold,
        )

        with st.spinner("Parsing input e query sul grafo..."):
            result = attributor.attribute(claim_input.strip())

        mode_label = "❓ Domanda" if result.is_question else "📝 Claim"
        st.caption(f"Modalità rilevata: **{mode_label}**")

        # ── Show parsed/resolved triple ───────────────────────────
        triple_header = "Risposta (tripla risolta dal grafo)" if result.is_question else "Parsed Triple"
        st.markdown(f"#### {triple_header}")
        if result.match_type == "parse_error":
            msg = result.source_chunk or "Impossibile parsare l'input."
            st.error(msg)
        elif result.match_type == "not_found" and not result.subject and not result.obj:
            st.warning("Nessuna tripla corrispondente trovata nel grafo.")
        else:
            answer_class = {
                "subject": ("triple-tag-answer", "triple-tag", "triple-tag"),
                "predicate": ("triple-tag", "triple-tag-answer", "triple-tag"),
                "object": ("triple-tag", "triple-tag", "triple-tag-answer"),
            }
            cs, cp, co = answer_class.get(
                result.answer_field if result.is_question else "",
                ("triple-tag", "triple-tag", "triple-tag"),
            )
            st.markdown(
                f'<span class="{cs}">S: {result.subject}</span> '
                f'<span class="{cp}">P: {result.predicate}</span> '
                f'<span class="{co}">O: {result.obj}</span>',
                unsafe_allow_html=True,
            )

        # ── Show attribution result ───────────────────────────────
        st.markdown("#### Verification Result")

        if result.match_type == "exact":
            st.markdown(
                f"""<div class="result-exact">
                    <strong>✅ EXACT MATCH</strong><br>
                    Triple found verbatim in the knowledge graph.
                </div>""",
                unsafe_allow_html=True,
            )
        elif result.match_type == "semantic":
            st.markdown(
                f"""<div class="result-semantic">
                    <strong>🟡 SEMANTIC MATCH</strong> — similarity: {result.similarity:.3f}<br>
                    Predicate matched via cosine similarity (threshold: {semantic_threshold}).
                </div>""",
                unsafe_allow_html=True,
            )
        elif result.match_type == "not_found":
            sim_str = f" (best similarity: {result.similarity:.3f})" if result.similarity > 0 else ""
            st.markdown(
                f"""<div class="result-notfound">
                    <strong>❌ NOT FOUND</strong>{sim_str}<br>
                    No matching triple in graph — claim cannot be attributed.
                </div>""",
                unsafe_allow_html=True,
            )
        elif result.match_type == "parse_error":
            pass  # already shown above

        # ── Source evidence ───────────────────────────────────────
        if result.source_chunk and result.match_type != "parse_error":
            st.markdown("#### Source Evidence")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f'<div class="chunk-box">{result.source_chunk}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""<div class="card">
                        <h4>Metadata</h4>
                        <p>
                            <strong>File:</strong> {result.source_file}<br>
                            <strong>Chunk:</strong> #{result.chunk_index}
                        </p>
                    </div>""",
                    unsafe_allow_html=True,
                )
