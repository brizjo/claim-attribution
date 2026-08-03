"""
Claim Attribution — LPG/Neo4j Streamlit App.

Tab 1 — Ingest ALCE: domanda ASQA → 5 passaggi → coref → triple
        (REBEL e/o DeepSeek) → Neo4j.  Corpus ESCLUSIVAMENTE ALCE:
        l'ingestione di PDF/TXT è stata rimossa (2026-08-03).
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
def get_alce_loader():
    """Corpus ALCE caricato una sola volta per sessione (~10MB)."""
    from src.ingestion.alce_loader import AlceLoader
    return AlceLoader()


@st.cache_resource
def get_ingestor(extractor_name: str):
    """Un ingestor per estrattore — il modello REBEL resta caricato."""
    from src.ingestion.alce_ingestor import AlceIngestor, build_extractor
    return AlceIngestor(
        client=get_neo4j_client(),
        extractor=build_extractor(extractor_name),
    )


@st.cache_resource
def get_deepseek():
    from src.ingestion.deepseek_extractor import DeepSeekExtractor
    return DeepSeekExtractor()


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
    st.markdown("#### Grafo per estrattore")
    if neo4j and neo4j.is_connected():
        rows = neo4j.stats_by_extractor()
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.columns = ["Extractor", "Archi", "Passaggi"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("Nessun arco nel grafo.")
    else:
        st.caption("Neo4j not connected.")

    st.markdown("#### Copertura estrattori (registro)")
    from src.ingestion.processed_registry import ProcessedRegistry
    _registry = ProcessedRegistry()
    _registry.reload()
    for _ext in settings.AVAILABLE_EXTRACTORS:
        _s = _registry.stats(_ext)
        if _s["docs"]:
            st.caption(
                f"`{_ext}` — {_s['docs']} doc, {_s['triples']} triple, "
                f"**{_s['zero_triple_docs']} a zero triple** "
                f"(copertura {_s['coverage']:.0%})"
            )
        else:
            st.caption(f"`{_ext}` — nessun documento processato")

    st.markdown("---")
    st.markdown("#### DeepSeek (unico LLM: estrazione + parsing domande)")
    _ds = get_deepseek()
    if _ds.is_available():
        st.markdown('<span class="badge-online">● API key configurata</span>', unsafe_allow_html=True)
        st.caption(f"Model: **{_ds.model}** — temperature {settings.DEEPSEEK_TEMPERATURE}")
    else:
        st.markdown('<span class="badge-offline">● API key mancante</span>', unsafe_allow_html=True)
        st.caption("Crea `.env` nella root con `DEEPSEEK_API_KEY=sk-...`")

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

tab_ingest, tab_claim = st.tabs(["📚 Corpus ALCE", "🔍 Claim Attribution"])


# ──────────────────────────────────────────────────────────────────────
# TAB 1 — INGEST ALCE
# ──────────────────────────────────────────────────────────────────────

def _render_triples(rows: list[dict], empty_msg: str) -> None:
    """Lista ⟨S,P,O⟩ + claim_span. `rows`: dict subject/predicate/object."""
    if not rows:
        st.caption(empty_msg)
        return
    for r in rows:
        st.markdown(
            f'<span class="triple-tag">S: {r["subject"]}</span> '
            f'<span class="triple-tag">P: {r["predicate"]}</span> '
            f'<span class="triple-tag">O: {r["object"]}</span>',
            unsafe_allow_html=True,
        )
        span = (r.get("claim_span") or "").strip()
        if span:
            st.caption(f"↳ {span}")


def _render_ingest_tab() -> None:
    """Corpo del tab ALCE — funzione così i `return` non fermano l'app intera."""
    st.markdown("### Corpus ALCE / ASQA")
    st.markdown(
        "<p style='color:#94a3b8;font-size:.9rem;'>"
        "Unica sorgente del sistema. Ogni domanda porta i primi 5 passaggi "
        "(già ri-rankati oracle, ~100 parole ciascuno = chunk nativi). "
        "Pipeline: testo originale → coref → estrazione triple "
        "(<strong>REBEL</strong> e/o <strong>DeepSeek</strong>) → Neo4j, "
        "con <code>source_id</code> = <code>doc[\"id\"]</code> come provenienza."
        "</p>",
        unsafe_allow_html=True,
    )

    loader = get_alce_loader()

    if not loader.exists():
        st.error(
            f"Corpus non trovato: `{loader.path}`\n\n"
            "Imposta `ALCE_DATA_PATH` in `config/settings.py` (o come variabile "
            "d'ambiente) sul file `asqa_eval_gtr_top100_reranked_oracle.json`."
        )
        return

    entries = loader.entries()

    # ── Selezione estrattori ──────────────────────────────────────────
    col_ext, col_search = st.columns([1, 2])
    with col_ext:
        active_extractors = st.multiselect(
            "Estrattori",
            options=settings.AVAILABLE_EXTRACTORS,
            default=[settings.EXTRACTOR_REBEL],
            help="I grafi restano separati: ogni arco porta la proprietà `extractor`.",
        )
    with col_search:
        search_q = st.text_input(
            "Filtra domande",
            placeholder="es. 'world cup', 'president', 'album'...",
        )

    if settings.EXTRACTOR_DEEPSEEK in active_extractors and not get_deepseek().is_available():
        st.warning(
            "DeepSeek selezionato ma `DEEPSEEK_API_KEY` non è configurata — "
            "crea `.env` nella root del progetto (vedi `.env.example`)."
        )

    # ── Stato di ingestione (Neo4j DISTINCT source_id ∪ registro) ─────
    status_ext = active_extractors[0] if active_extractors else settings.EXTRACTOR_REBEL
    processed_ids: set[str] = set()
    if neo4j is not None:
        try:
            processed_ids = get_ingestor(status_ext).processed_ids()
        except Exception as exc:
            st.warning(f"Stato ingestione non disponibile: {exc}")

    filtered = loader.search(search_q, limit=200)

    def _entry_label(e) -> str:
        ids = {d["source_id"] for d in e.docs()}
        done = ids & processed_ids
        mark = "✓" if done and len(done) == len(ids) else ("◐" if done else "○")
        return f"{mark}  {e.question}"

    if not filtered:
        st.info("Nessuna domanda corrisponde al filtro.")
        return

    st.caption(
        f"{len(filtered)} domande mostrate su {len(entries)} — "
        f"✓ tutti i passaggi ingeriti con `{status_ext}`, ◐ parziale, ○ nessuno"
    )

    selected = st.selectbox(
        "Domanda",
        options=filtered,
        format_func=_entry_label,
        index=0,
    )

    # ── Ground truth ASQA ─────────────────────────────────────────────
    st.markdown(f"#### {selected.question}")
    st.caption(f"sample_id: `{selected.sample_id}`")
    with st.expander(f"Ground truth — {len(selected.qa_pairs)} sotto-risposte", expanded=False):
        for i, qa in enumerate(selected.qa_pairs):
            answers = ", ".join(qa.get("short_answers", []))
            st.markdown(f"**{i}.** {qa.get('question', '')} → *{answers}*")

    # ── Azioni ────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        ingest_btn = st.button(
            "⚙️ Ingerisci questa domanda",
            type="primary",
            use_container_width=True,
            disabled=(neo4j is None or not active_extractors),
        )
    with col_b:
        force = st.checkbox("Force re-ingest", value=False, help="Cancella gli archi esistenti di questo passaggio e riestrae.")
    with col_c:
        clear_btn = st.button("🗑️ Clear Graph", use_container_width=True, disabled=(neo4j is None))

    if clear_btn and neo4j:
        neo4j.clear_graph()
        st.cache_resource.clear()
        st.success("Graph cleared. Il registro `processed_ids.txt` NON è stato toccato: "
                   "usa 'Force re-ingest' per riprocessare.")
        st.rerun()

    # ── Ingestione ────────────────────────────────────────────────────
    if ingest_btn and neo4j:
        reports = {}
        for ext_name in active_extractors:
            with st.status(f"🌀 {ext_name}: ingestione di 5 passaggi...", expanded=True) as stage:
                try:
                    ingestor = get_ingestor(ext_name)
                except Exception as exc:
                    stage.update(label=f"❌ {ext_name}: {exc}", state="error")
                    continue

                report = ingestor.ingest_entry(
                    selected,
                    skip_existing=not force,
                    force=force,
                    progress=lambda msg: stage.update(label=f"🌀 {msg}"),
                )
                reports[ext_name] = report

                zero = len(report.zero_triple_docs)
                skipped = sum(1 for d in report.docs if d.skipped)
                errors = [d for d in report.docs if d.error]
                stage.update(
                    label=(
                        f"✅ {ext_name}: {report.total_triples} triple "
                        f"({len(report.processed)} passaggi processati, "
                        f"{skipped} saltati, {zero} a zero triple, "
                        f"{len(errors)} errori)"
                    ),
                    state="error" if errors else "complete",
                )
                for d in errors:
                    st.error(f"{ext_name} / {d.source_id}: {d.error}")

        # Testo coref-risolto: solo in memoria, non persistito nel grafo.
        st.session_state["last_reports"] = {
            ext: {d.source_id: d for d in rep.docs} for ext, rep in reports.items()
        }
        st.session_state["last_sample_id"] = selected.sample_id

    # ── Vista corpus + triple ─────────────────────────────────────────
    st.markdown("### Passaggi e triple estratte")

    last = (
        st.session_state.get("last_reports", {})
        if st.session_state.get("last_sample_id") == selected.sample_id
        else {}
    )

    for chunk in selected.docs():
        sid = chunk["source_id"]
        found = chunk.get("answers_found") or []
        supports = [str(i) for i, v in enumerate(found) if v]
        badge = f" — supporta le sotto-risposte {', '.join(supports)}" if supports else ""

        with st.expander(
            f"[{chunk['chunk_index']}] {chunk['title']}  ·  source_id `{sid}`{badge}",
            expanded=False,
        ):
            st.markdown("**Testo originale (corpus ALCE)**")
            st.markdown(f'<div class="chunk-box">{chunk["text"]}</div>', unsafe_allow_html=True)

            for ext_name in settings.AVAILABLE_EXTRACTORS:
                doc_result = last.get(ext_name, {}).get(sid)
                if doc_result and doc_result.resolved_text:
                    st.markdown(f"**Testo coref-risolto — input di `{ext_name}`**")
                    st.markdown(
                        f'<div class="chunk-box">{doc_result.resolved_text}</div>',
                        unsafe_allow_html=True,
                    )
                    break  # il coref non dipende dall'estrattore: uno basta

            col_r, col_d = st.columns(2)
            for col, ext_name in zip((col_r, col_d), settings.AVAILABLE_EXTRACTORS):
                with col:
                    st.markdown(f"**Triple — `{ext_name}`**")
                    rows = []
                    if neo4j is not None:
                        try:
                            rows = neo4j.triples_by_source(sid, extractor=ext_name)
                        except Exception as exc:
                            st.caption(f"Neo4j: {exc}")
                    doc_result = last.get(ext_name, {}).get(sid)
                    if not rows and doc_result:
                        # Fallback: run appena concluso ma grafo non interrogabile.
                        rows = [
                            {"subject": t.subject, "predicate": t.predicate,
                             "object": t.obj, "claim_span": t.claim_span}
                            for t in doc_result.triples
                        ]
                    msg = "— nessuna tripla estratta"
                    if doc_result and doc_result.skipped:
                        msg = "— già processato (nessuna tripla nel grafo)"
                    elif doc_result and doc_result.error:
                        msg = f"— errore: {doc_result.error}"
                    _render_triples(rows, msg)


with tab_ingest:
    _render_ingest_tab()


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

    # Il filtro estrattore si applica a TUTTE le query di attribution:
    # i grafi rebel/deepseek non vanno mai interrogati insieme.
    query_extractor = st.radio(
        "Grafo da interrogare (estrattore)",
        options=settings.AVAILABLE_EXTRACTORS,
        index=settings.AVAILABLE_EXTRACTORS.index(settings.ACTIVE_EXTRACTOR)
        if settings.ACTIVE_EXTRACTOR in settings.AVAILABLE_EXTRACTORS else 0,
        horizontal=True,
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
            extractor=query_extractor,
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
                            <strong>Titolo:</strong> {result.source_file}<br>
                            <strong>source_id:</strong> {result.source_id or "n/a"}<br>
                            <strong>Estrattore:</strong> {query_extractor}<br>
                            <strong>Passaggio:</strong> #{result.chunk_index}
                        </p>
                    </div>""",
                    unsafe_allow_html=True,
                )
