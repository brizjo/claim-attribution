"""
Claim Attribution -- LPG/Neo4j Streamlit App.

Tab 1 -- Corpus ALCE: 6-step pipeline
    Step 1: Choose ASQA question
    Step 2: Show passages
    Step 3: Coreference resolution
    Step 4: Extract triples (REBEL / DeepSeek)
    Step 5: Review all triples
    Step 6: Write to Neo4j

Tab 2 -- Claim Attribution: input claim -> exact match / semantic fallback -> source

Run: streamlit run app.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ["HF_HOME"] = r"D:\hf_home"
os.environ["HF_HUB_CACHE"] = r"D:\hf_home\hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\hf_home\hub"
# TRANSFORMERS_CACHE deliberately NOT set — see config/settings.py comment:
# it's a second cache root separate from HF_HUB_CACHE, and a stale/incomplete
# blob there gets re-downloaded every session instead of using the good copy.
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"D:\hf_home\sentence_transformers"
os.environ["HF_DATASETS_CACHE"] = r"D:\hf_home\datasets"
os.environ["TORCH_HOME"] = r"D:\hf_home\torch"

import streamlit as st

from config import settings

st.set_page_config(
    page_title="Claim Attribution",
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
.step-header {
    font-size:1.1rem; font-weight:600; color:#a5b4fc;
    border-bottom:1px solid rgba(99,102,241,.2); padding-bottom:.4rem;
    margin-top:1.5rem; margin-bottom:.8rem; }
.span-label {
    color:#94a3b8; font-size:.82rem; font-style:italic;
    margin-top:4px; margin-bottom:8px; }
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
    except Exception:
        return None


@st.cache_resource
def get_alce_loader():
    """ALCE corpus loaded once per session (~10MB)."""
    from src.ingestion.alce_loader import AlceLoader
    return AlceLoader()


@st.cache_resource
def get_ingestor(extractor_name: str):
    """One ingestor per extractor — keeps REBEL model loaded."""
    from src.ingestion.alce_ingestor import AlceIngestor, build_extractor
    return AlceIngestor(
        client=get_neo4j_client(),
        extractor=build_extractor(extractor_name),
    )


@st.cache_resource
def get_deepseek():
    from src.ingestion.deepseek_extractor import DeepSeekExtractor
    return DeepSeekExtractor()


@st.cache_resource
def get_debug_rebel_extractor():
    """Standalone REBEL — no Neo4j dependency, for debug/timing."""
    from src.ingestion.triple_extractor import TripleExtractor
    return TripleExtractor()


@st.cache_resource
def get_debug_coref_resolver():
    from src.ingestion.coref_resolver import CoreferenceResolver
    return CoreferenceResolver()


@st.cache_resource
def get_sentence_splitter():
    """spaCy sentence splitter — usato dalla pipeline ibrida (REBEL per frase)."""
    from src.segmentation.sentence_splitter import SentenceSplitter
    return SentenceSplitter()


@st.cache_resource
def get_hybrid_extractor(variant: str):
    """Un HybridExtractor per variante — riusa REBEL/DeepSeek/spaCy gia' caricati."""
    from src.ingestion.hybrid_extractor import HybridExtractor
    return HybridExtractor(
        rebel=get_debug_rebel_extractor(),
        deepseek_client=get_deepseek().client,
        splitter=get_sentence_splitter(),
        variant=variant,
    )


@st.cache_resource
def get_graph_writer():
    """One GraphWriter — keeps its SentenceTransformer loaded across writes."""
    from src.ingestion.graph_writer import GraphWriter
    return GraphWriter(client=get_neo4j_client())


@st.cache_resource
def get_attributor(semantic_threshold: float, extractor: str):
    """One ClaimAttributor per (threshold, extractor) — keeps REBEL + encoder loaded."""
    from src.attribution.claim_attributor import ClaimAttributor
    return ClaimAttributor(
        client=get_neo4j_client(),
        semantic_threshold=semantic_threshold,
        extractor=extractor,
    )


# ====================================================================
# Header
# ====================================================================

st.markdown("""
<div class="main-header">
    <h1>Claim Attribution -- LPG/Neo4j</h1>
</div>
""", unsafe_allow_html=True)


# ====================================================================
# Sidebar — status
# ====================================================================

with st.sidebar:
    st.markdown("### System Status")

    neo4j = get_neo4j_client()
    if neo4j and neo4j.is_connected():
        st.markdown('<span class="badge-online">Neo4j Connected</span>', unsafe_allow_html=True)
        stats = neo4j.stats()
        st.caption(f"Entities: **{stats['nodes']}** | Relations: **{stats['relations']}**")
        active_db = getattr(neo4j, "database", None)
        if active_db:
            st.caption(f"Active DB: `{active_db}` -- make sure Browser points here")
        list_dbs = getattr(neo4j, "list_databases", None)
        if callable(list_dbs):
            dbs = list_dbs()
            if dbs:
                st.caption(f"Visible DBs: {', '.join(dbs)}")
        if st.button("Reload Neo4j Client"):
            get_neo4j_client.clear()
            st.rerun()
    else:
        st.markdown('<span class="badge-offline">Neo4j Offline</span>', unsafe_allow_html=True)
        st.warning(
            "Start Neo4j Desktop, open a database, then set:\n"
            "```\nNEO4J_PASSWORD=yourpass\n```\nin your environment."
        )
        if st.button("Retry Connection"):
            get_neo4j_client.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("#### Graph by extractor")
    if neo4j and neo4j.is_connected():
        rows = neo4j.stats_by_extractor()
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.columns = ["Extractor", "Edges", "Passages"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No edges in the graph.")
    else:
        st.caption("Neo4j not connected.")

    st.markdown("#### Extractor coverage (registry)")
    from src.ingestion.processed_registry import ProcessedRegistry
    _registry = ProcessedRegistry()
    _registry.reload()
    for _ext in settings.AVAILABLE_EXTRACTORS:
        _s = _registry.stats(_ext)
        if _s["docs"]:
            st.caption(
                f"`{_ext}` -- {_s['docs']} docs, {_s['triples']} triples, "
                f"**{_s['zero_triple_docs']} with zero triples** "
                f"(coverage {_s['coverage']:.0%})"
            )
        else:
            st.caption(f"`{_ext}` -- no documents processed")

    st.markdown("---")
    st.markdown("#### DeepSeek (LLM: extraction + question parsing)")
    _ds = get_deepseek()
    if _ds.is_available():
        st.markdown('<span class="badge-online">API key configured</span>', unsafe_allow_html=True)
        st.caption(f"Model: **{_ds.model}** -- temperature {settings.DEEPSEEK_TEMPERATURE}")
    else:
        st.markdown('<span class="badge-offline">API key missing</span>', unsafe_allow_html=True)
        st.caption("Create `.env` in project root with `DEEPSEEK_API_KEY=sk-...`")

    st.markdown("---")
    st.markdown("#### Settings")
    semantic_threshold = st.slider(
        "Semantic similarity threshold",
        0.5, 1.0, settings.SEMANTIC_THRESHOLD, 0.05,
    )
    st.markdown("---")
    st.caption("Claim Attribution v1.0 -- LPG/Neo4j")


# ====================================================================
# Tabs
# ====================================================================

tab_ingest, tab_hybrid, tab_claim = st.tabs(
    ["Corpus ALCE", "Hybrid Debug (REBEL + DeepSeek)", "Claim Attribution"]
)


# ──────────────────────────────────────────────────────────────────────
# TAB 1 — CORPUS ALCE (6-step pipeline)
# ──────────────────────────────────────────────────────────────────────

def _render_triples(rows: list[dict], empty_msg: str) -> None:
    """Render a list of (S, P, O) triples with claim_span."""
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
            st.markdown(
                f'<div class="span-label">source: {span}</div>',
                unsafe_allow_html=True,
            )


def _render_ingest_tab() -> None:
    """Body of the ALCE tab — function so `return` does not stop the whole app."""

    st.markdown("### Corpus ALCE / ASQA")
    st.markdown(
        "<p style='color:#94a3b8;font-size:.9rem;'>"
        "Sole data source. Each question has 5 passages "
        "(oracle-reranked, ~100 words each = native chunks). "
        "Pipeline: original text -> coref -> triple extraction "
        "(<strong>REBEL</strong> and/or <strong>DeepSeek</strong>) -> Neo4j, "
        "with <code>source_id</code> = <code>doc[\"id\"]</code> as provenance."
        "</p>",
        unsafe_allow_html=True,
    )

    loader = get_alce_loader()

    if not loader.exists():
        st.error(
            f"Corpus not found: `{loader.path}`\n\n"
            "Set `ALCE_DATA_PATH` in `config/settings.py` (or as an "
            "environment variable) to the `asqa_eval_gtr_top100_reranked_oracle.json` file."
        )
        return

    entries = loader.entries()

    # ================================================================
    # STEP 1 — Choose Question
    # ================================================================
    st.markdown('<div class="step-header">Step 1 -- Choose Question</div>', unsafe_allow_html=True)

    col_ext, col_search = st.columns([1, 2])
    with col_ext:
        active_extractor = st.radio(
            "Extractor",
            options=settings.AVAILABLE_EXTRACTORS,
            index=0,
            horizontal=True,
            help="Select which extractor to use for triple extraction.",
        )
    with col_search:
        search_q = st.text_input(
            "Filter questions",
            placeholder="e.g. 'world cup', 'president', 'album'...",
        )

    if active_extractor == settings.EXTRACTOR_DEEPSEEK and not get_deepseek().is_available():
        st.warning(
            "DeepSeek selected but `DEEPSEEK_API_KEY` is not configured -- "
            "create `.env` in project root (see `.env.example`)."
        )

    # Ingestion status (Neo4j DISTINCT source_id + registry)
    processed_ids: set[str] = set()
    if neo4j is not None:
        try:
            processed_ids = get_ingestor(active_extractor).processed_ids()
        except Exception as exc:
            st.warning(f"Ingestion status unavailable: {exc}")

    filtered = loader.search(search_q, limit=200)

    def _entry_label(e) -> str:
        ids = {d["source_id"] for d in e.docs()}
        done = ids & processed_ids
        if done and len(done) == len(ids):
            mark = "[done]"
        elif done:
            mark = "[partial]"
        else:
            mark = "[new]"
        return f"{mark}  {e.question}"

    if not filtered:
        st.info("No questions match the filter.")
        return

    st.caption(
        f"{len(filtered)} questions shown out of {len(entries)} -- "
        f"[done] = all passages ingested with `{active_extractor}`, "
        f"[partial] = some, [new] = none"
    )

    selected = st.selectbox(
        "Question",
        options=filtered,
        format_func=_entry_label,
        index=0,
    )

    # Ground truth
    st.markdown(f"#### {selected.question}")
    st.caption(f"sample_id: `{selected.sample_id}`")
    with st.expander(f"Ground truth -- {len(selected.qa_pairs)} sub-answers", expanded=False):
        for i, qa in enumerate(selected.qa_pairs):
            answers = ", ".join(qa.get("short_answers", []))
            st.markdown(f"**{i}.** {qa.get('question', '')} -> *{answers}*")

    # ================================================================
    # STEP 2 — Show Passages
    # ================================================================
    st.markdown('<div class="step-header">Step 2 -- Show Passages</div>', unsafe_allow_html=True)

    for chunk in selected.docs():
        sid = chunk["source_id"]
        found = chunk.get("answers_found") or []
        supports = [str(i) for i, v in enumerate(found) if v]
        badge = f" -- supports sub-answers {', '.join(supports)}" if supports else ""

        with st.expander(
            f"[{chunk['chunk_index']}] {chunk['title']}  |  source_id `{sid}`{badge}",
            expanded=False,
        ):
            st.markdown("**Original text (ALCE corpus)**")
            st.markdown(f'<div class="chunk-box">{chunk["text"]}</div>', unsafe_allow_html=True)

    # ================================================================
    # STEP 3 — Coreference Resolution
    # ================================================================
    st.markdown('<div class="step-header">Step 3 -- Coreference Resolution</div>', unsafe_allow_html=True)

    skip_coref = st.checkbox("Skip coreference resolution", value=False,
                              help="Disable coref if you want raw text passed to the extractor.")

    # ================================================================
    # STEP 4 — Extract Triples
    # ================================================================
    st.markdown('<div class="step-header">Step 4 -- Extract Triples</div>', unsafe_allow_html=True)

    extract_btn = st.button(
        "Extract Triples",
        type="primary",
        use_container_width=True,
    )

    if extract_btn:
        import time
        from src.ingestion.span_matcher import best_span
        from src.ingestion.output_store import save_coref

        extractor_obj = get_debug_rebel_extractor() if active_extractor == settings.EXTRACTOR_REBEL else get_deepseek()
        resolver = get_debug_coref_resolver()
        results = []

        with st.status(f"{active_extractor}: extracting triples...", expanded=True) as stage:
            for chunk in selected.docs():
                original = chunk.get("text", "")
                sid = chunk["source_id"]
                stage.update(label=f"Processing source_id={sid}...")

                # Coref
                if not skip_coref:
                    resolved = resolver.resolve(original)
                else:
                    resolved = original

                # Save coref to JSONL
                save_coref(
                    source_id=sid,
                    sample_id=selected.sample_id,
                    title=chunk.get("title", ""),
                    chunk_index=chunk.get("chunk_index", 0),
                    original_text=original,
                    resolved_text=resolved,
                )

                # Extract
                t0 = time.time()
                raw = extractor_obj.extract([{**chunk, "text": resolved}])
                elapsed = time.time() - t0

                triples = [
                    t._replace(chunk_text=original, claim_span=best_span(original, t.subject, t.obj))
                    for t in raw
                ]

                results.append({
                    "source_id": sid,
                    "chunk_index": chunk.get("chunk_index", 0),
                    "title": chunk.get("title", ""),
                    "original_text": original,
                    "resolved_text": resolved,
                    "triples": triples,
                    "elapsed": elapsed,
                })

            total = sum(len(r["triples"]) for r in results)
            total_t = sum(r["elapsed"] for r in results)
            stage.update(
                label=f"Done: {total} triples total, {total_t:.3f}s ({active_extractor})",
                state="complete",
            )

        # Save triples to JSONL
        from src.ingestion.output_store import save_triples_batch
        for r in results:
            if r["triples"]:
                triples_dicts = [
                    {
                        "source_id": r["source_id"],
                        "subject": t.subject,
                        "predicate": t.predicate,
                        "obj": t.obj,
                        "claim_span": t.claim_span,
                        "chunk_text": t.chunk_text,
                        "source_file": t.source_file,
                        "title": r["title"],
                        "chunk_index": t.chunk_index,
                    }
                    for t in r["triples"]
                ]
                save_triples_batch(triples_dicts, selected.sample_id, active_extractor)

        st.session_state["extract_results"] = {
            "sample_id": selected.sample_id,
            "extractor": active_extractor,
            "results": results,
        }

    # ================================================================
    # STEP 5 — Review All Triples
    # ================================================================
    st.markdown('<div class="step-header">Step 5 -- Review All Triples</div>', unsafe_allow_html=True)

    extract_data = st.session_state.get("extract_results")
    if extract_data and extract_data["sample_id"] == selected.sample_id:
        ext_label = extract_data["extractor"]
        total_triples = sum(len(r["triples"]) for r in extract_data["results"])
        total_time = sum(r["elapsed"] for r in extract_data["results"])
        st.caption(
            f"Extractor: `{ext_label}` | "
            f"Total triples: {total_triples} | "
            f"Total time: {total_time:.3f}s"
        )

        for r in extract_data["results"]:
            with st.expander(
                f"[{r['chunk_index']}] {r['title']}  |  source_id `{r['source_id']}`  |  "
                f"time: {r['elapsed']:.3f}s  |  {len(r['triples'])} triples",
                expanded=False,
            ):
                st.markdown("**Original text (ALCE passage)**")
                st.markdown(f'<div class="chunk-box">{r["original_text"]}</div>', unsafe_allow_html=True)

                if r["resolved_text"] != r["original_text"]:
                    st.markdown(f"**Coref-resolved text -- input to `{ext_label}`**")
                    st.markdown(f'<div class="chunk-box">{r["resolved_text"]}</div>', unsafe_allow_html=True)

                st.markdown("**Extracted triples**")
                rows = [
                    {"subject": t.subject, "predicate": t.predicate, "object": t.obj, "claim_span": t.claim_span}
                    for t in r["triples"]
                ]
                _render_triples(rows, "-- no triples extracted")
    else:
        st.caption("No extraction results yet. Run Step 4 first.")

    # ================================================================
    # STEP 6 — Write to Neo4j
    # ================================================================
    st.markdown('<div class="step-header">Step 6 -- Write to Neo4j</div>', unsafe_allow_html=True)

    col_w, col_f, col_c = st.columns([2, 1, 1])
    with col_w:
        write_btn = st.button(
            "Write to Neo4j",
            type="primary",
            use_container_width=True,
            disabled=(
                neo4j is None
                or not extract_data
                or extract_data.get("sample_id") != selected.sample_id
            ),
        )
    with col_f:
        force_write = st.checkbox("Force re-write", value=False,
                                   help="Delete existing edges for these passages and re-write.")
    with col_c:
        clear_btn = st.button("Clear Graph", use_container_width=True, disabled=(neo4j is None))

    if clear_btn and neo4j:
        neo4j.clear_graph()
        st.cache_resource.clear()
        st.success("Graph cleared. The registry `processed_ids.txt` was NOT touched: "
                   "use 'Force re-write' to reprocess.")
        st.rerun()

    if write_btn and neo4j and extract_data:
        from src.ingestion.processed_registry import ProcessedRegistry
        from src.ingestion.output_store import save_ingest_report

        writer = get_graph_writer()
        registry = ProcessedRegistry()
        ext_name = extract_data["extractor"]
        total_written = 0
        errors = []

        with st.status(f"Writing triples to Neo4j ({ext_name})...", expanded=True) as stage:
            for r in extract_data["results"]:
                sid = r["source_id"]
                triples = r["triples"]

                if force_write:
                    neo4j.delete_by_source(sid, ext_name)

                if not triples:
                    registry.mark(sid, ext_name, 0)
                    continue

                try:
                    stage.update(label=f"Writing {len(triples)} triples for source_id={sid}...")
                    written = writer.write_triples(triples)
                    total_written += written
                    registry.mark(sid, ext_name, written)
                except Exception as exc:
                    errors.append(f"{sid}: {exc}")
                    stage.update(label=f"Error on {sid}: {exc}", state="error")

            stage.update(
                label=f"Done: {total_written} triples written to Neo4j ({ext_name})",
                state="error" if errors else "complete",
            )

        # Save ingest report to JSONL
        save_ingest_report(
            sample_id=selected.sample_id,
            question=selected.question,
            extractor=ext_name,
            total_triples=total_written,
            docs_processed=len(extract_data["results"]),
            docs_skipped=0,
            zero_triple_docs=sum(1 for r in extract_data["results"] if not r["triples"]),
            errors=errors,
        )

        for err in errors:
            st.error(err)

    # Show existing graph triples for this question
    if neo4j and neo4j.is_connected():
        with st.expander("Graph triples for this question (from Neo4j)", expanded=False):
            for chunk in selected.docs():
                sid = chunk["source_id"]
                try:
                    rows = neo4j.triples_by_source(sid, extractor=active_extractor)
                except Exception:
                    rows = []
                if rows:
                    st.markdown(f"**source_id `{sid}` -- {chunk.get('title', '')}**")
                    _render_triples(rows, "")


with tab_ingest:
    _render_ingest_tab()


# ──────────────────────────────────────────────────────────────────────
# TAB 2 — HYBRID DEBUG (REBEL + DeepSeek, varianti A / B)
# ──────────────────────────────────────────────────────────────────────

def _triple_line(subject: str, predicate: str, obj: str, extra: str = "") -> None:
    st.markdown(
        f'<span class="triple-tag">S: {subject}</span> '
        f'<span class="triple-tag">P: {predicate}</span> '
        f'<span class="triple-tag">O: {obj}</span>'
        + (f' <span class="span-label">{extra}</span>' if extra else ""),
        unsafe_allow_html=True,
    )


def _passage_rows(report) -> list[dict]:
    """Righe della tabella di debug: una per passaggio + totali."""
    rows = []
    for p in report.passages:
        rows.append({
            "passage": f"[{p.chunk_index}] {p.source_id}",
            "triple prodotte": p.produced,
            "sopravvissute": len(p.survived),
            "scartate (guardrail)": len(p.discarded),
            "REBEL prodotte": len(p.rebel_raw),
            "REBEL confermate": p.rebel_matched,
            "REBEL rigettate": len(p.rebel_rejected),
            "triple finali da REBEL": p.rebel_kept,
            "LLM calls": p.llm_calls,
            "sec": round(p.rebel_seconds + p.llm_seconds, 2),
        })
    rows.append({
        "passage": "TOTALE",
        "triple prodotte": report.produced,
        "sopravvissute": report.survived,
        "scartate (guardrail)": report.produced - report.survived,
        "REBEL prodotte": report.rebel_produced,
        "REBEL confermate": report.rebel_matched,
        "REBEL rigettate": report.rebel_rejected,
        "triple finali da REBEL": report.rebel_kept,
        "LLM calls": report.llm_calls,
        "sec": round(report.seconds, 2),
    })
    return rows


def _render_report(report) -> None:
    import pandas as pd
    from src.ingestion.hybrid_extractor import VARIANT_LABELS, ORIGIN_REBEL_CONFIRMED

    st.markdown(f"#### Variante {VARIANT_LABELS[report.variant]}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Triple prodotte", report.produced)
    c2.metric("Sopravvissute ai guardrail", report.survived)
    c3.metric("REBEL rigettate", f"{report.rebel_rejected} / {report.rebel_produced}")
    c4.metric("Contributo REBEL", f"{report.rebel_kept} triple finali",
              help="Triple finali nate da un candidato REBEL. Piu' candidati REBEL "
                   "con lo stesso (subject, object) collassano in una sola tripla, "
                   "quindi e' <= 'REBEL confermate'.")

    st.dataframe(pd.DataFrame(_passage_rows(report)),
                 use_container_width=True, hide_index=True)

    reasons = report.discard_reasons
    if reasons:
        st.caption("Motivi di scarto (guardrail)")
        st.dataframe(
            pd.DataFrame([{"motivo": k, "triple": v} for k, v in reasons.most_common()]),
            use_container_width=True, hide_index=True,
        )

    for err in report.errors:
        st.error(err)

    for p in report.passages:
        with st.expander(
            f"[{p.chunk_index}] {p.title} | `{p.source_id}` | "
            f"{p.produced} prodotte -> {len(p.survived)} sopravvissute | "
            f"REBEL {len(p.rebel_raw)} -> rigettate {len(p.rebel_rejected)}",
            expanded=False,
        ):
            st.markdown("**Testo originale (evidenza verbatim)**")
            st.markdown(f'<div class="chunk-box">{p.original_text}</div>',
                        unsafe_allow_html=True)
            if p.resolved_text != p.original_text:
                st.markdown("**Testo coref-risolto (input ai modelli)**")
                st.markdown(f'<div class="chunk-box">{p.resolved_text}</div>',
                            unsafe_allow_html=True)
            st.caption(f"{len(p.sentences)} frasi -> REBEL ({p.rebel_seconds:.2f}s) "
                       f"| DeepSeek {p.llm_calls} call ({p.llm_seconds:.2f}s)")

            st.markdown("**Triple REBEL grezze**")
            if p.rebel_raw:
                for t in p.rebel_raw:
                    _triple_line(t["subject"], t["predicate"], t["obj"])
            else:
                st.caption("-- nessuna tripla REBEL")

            if p.deepseek_first_pass:
                st.markdown("**DeepSeek passata 1 (senza vedere REBEL)**")
                for t in p.deepseek_first_pass:
                    _triple_line(t["subject"], t["predicate"], t["obj"])

            st.markdown(f"**Triple finali sopravvissute ({len(p.survived)})**")
            if p.survived:
                for t in p.survived:
                    _triple_line(t.subject, t.predicate, t.obj, f"origin: {t.origin}")
                    st.markdown(f'<div class="span-label">span: {t.claim_span}</div>',
                                unsafe_allow_html=True)
            else:
                st.caption("-- nessuna tripla sopravvissuta")

            if p.discarded:
                st.markdown(f"**Scartate dai guardrail ({len(p.discarded)})**")
                for t in p.discarded:
                    _triple_line(t.subject, t.predicate, t.obj,
                                 f"scartata: {t.reason} | origin: {t.origin}")

            if p.rebel_rejected:
                st.markdown(f"**Triple REBEL rigettate ({len(p.rebel_rejected)})**")
                for r in p.rebel_rejected:
                    _triple_line(r.subject, r.predicate, r.obj, f"motivo: {r.reason}")


def _render_hybrid_tab() -> None:
    """Debug della pipeline ibrida: nessuna scrittura su Neo4j, solo JSONL + UI."""
    import pandas as pd
    from src.ingestion.hybrid_extractor import RunReport, VARIANTS, VARIANT_LABELS
    from src.ingestion.output_store import save_hybrid_report

    st.markdown("### Pipeline ibrida REBEL + DeepSeek")
    st.markdown(
        "<p style='color:#94a3b8;font-size:.9rem;'>"
        "Ordine fisso: coref -> sentence split -> estrazione. Due varianti a confronto:"
        "<br><strong>A</strong> - REBEL estrae, DeepSeek riceve passaggio + triple REBEL "
        "e produce il set finale (corregge, scarta, completa). 1 chiamata LLM/passaggio."
        "<br><strong>B</strong> - DeepSeek estrae da solo (senza vedere REBEL), poi REBEL "
        "estrae e DeepSeek fa da validatore sulle sue triple. 2 chiamate LLM/passaggio."
        "<br>Ogni tripla finale deve avere uno span <em>verbatim</em> del testo originale "
        "che contiene subject e object; i guardrail scartano predicato assente, "
        "subject == object, object = intera frase, span mancante."
        "</p>",
        unsafe_allow_html=True,
    )

    loader = get_alce_loader()
    if not loader.exists():
        st.error(f"Corpus non trovato: `{loader.path}`")
        return

    if not get_deepseek().is_available():
        st.warning("`DEEPSEEK_API_KEY` non configurata: la pipeline ibrida non puo' girare.")

    col_q, col_v = st.columns([2, 1])
    with col_q:
        search_q = st.text_input("Filtra domande", key="hyb_search",
                                 placeholder="es. 'world cup', 'president'...")
    with col_v:
        variants = st.multiselect(
            "Varianti da eseguire", options=list(VARIANTS), default=[VARIANTS[0]],
            format_func=lambda v: VARIANT_LABELS[v], key="hyb_variants",
        )

    filtered = loader.search(search_q, limit=200)
    if not filtered:
        st.info("Nessuna domanda corrisponde al filtro.")
        return

    selected = st.selectbox("Domanda", options=filtered,
                            format_func=lambda e: e.question, key="hyb_question")
    st.caption(f"sample_id: `{selected.sample_id}` -- {len(selected.docs())} passaggi")

    col_c, col_n, col_r = st.columns([1, 1, 2])
    with col_c:
        skip_coref = st.checkbox("Salta coref", value=False, key="hyb_skip_coref")
    with col_n:
        n_docs = st.number_input("Passaggi", min_value=1, max_value=len(selected.docs()),
                                 value=len(selected.docs()), key="hyb_n_docs")
    with col_r:
        run_btn = st.button("Esegui pipeline ibrida", type="primary",
                            use_container_width=True,
                            disabled=(not variants or not get_deepseek().is_available()))

    if run_btn:
        from src.ingestion.output_store import save_coref

        chunks = selected.docs()[:int(n_docs)]
        resolver = None if skip_coref else get_debug_coref_resolver()
        reports = {}

        with st.status("Pipeline ibrida in corso...", expanded=True) as stage:
            # Coref una volta sola: le varianti devono partire dallo stesso input.
            resolved_map = {}
            for chunk in chunks:
                sid = chunk["source_id"]
                original = chunk.get("text", "")
                stage.update(label=f"coref {sid}...")
                resolved = resolver.resolve(original) if resolver else original
                resolved_map[sid] = resolved
                save_coref(
                    source_id=sid, sample_id=selected.sample_id,
                    title=chunk.get("title", ""), chunk_index=chunk.get("chunk_index", 0),
                    original_text=original, resolved_text=resolved,
                )

            for variant in variants:
                extractor = get_hybrid_extractor(variant)
                report = RunReport(variant=variant, sample_id=selected.sample_id,
                                   question=selected.question)
                for chunk in chunks:
                    stage.update(label=f"variante {variant}: {chunk['source_id']}...")
                    report.passages.append(
                        extractor.run_passage(chunk, resolved_map[chunk["source_id"]])
                    )
                counts = save_hybrid_report(report)
                reports[variant] = report
                stage.update(label=f"variante {variant}: {counts['survived']} triple salvate")

            stage.update(label="Pipeline ibrida completata", state="complete")

        st.session_state["hybrid_reports"] = {
            "sample_id": selected.sample_id,
            "reports": reports,
        }

    data = st.session_state.get("hybrid_reports")
    if not data or data["sample_id"] != selected.sample_id:
        st.caption("Nessun risultato per questa domanda. Esegui la pipeline.")
        return

    reports = data["reports"]

    if len(reports) > 1:
        st.markdown("#### Confronto varianti")
        st.dataframe(pd.DataFrame([
            {
                "variante": v,
                "triple prodotte": r.produced,
                "sopravvissute": r.survived,
                "REBEL prodotte": r.rebel_produced,
                "REBEL confermate": r.rebel_matched,
                "REBEL rigettate": r.rebel_rejected,
                "triple finali da REBEL": r.rebel_kept,
                "LLM calls": r.llm_calls,
                "sec": round(r.seconds, 2),
            }
            for v, r in reports.items()
        ]), use_container_width=True, hide_index=True)

    for variant, report in reports.items():
        _render_report(report)

    st.caption(
        "Output su `data/outputs/`: `triples_hybrid.jsonl` (sopravvissute), "
        "`triples_hybrid_discarded.jsonl` (scartate + motivo), "
        "`hybrid_runs.jsonl` (statistiche di run). Nessuna scrittura su Neo4j."
    )


with tab_hybrid:
    _render_hybrid_tab()



# ──────────────────────────────────────────────────────────────────────
# TAB 2 — CLAIM ATTRIBUTION
# ──────────────────────────────────────────────────────────────────────

with tab_claim:
    st.markdown("### Verify Claim or Question")
    st.markdown(
        "<p style='color:#94a3b8;font-size:.9rem;'>"
        "Enter a statement (claim) <em>or</em> a question. "
        "Statements are parsed via mREBEL and verified against the graph. "
        "Questions are converted to a partial triple via LLM and resolved "
        "using pattern query + cosine similarity on the predicate."
        "</p>",
        unsafe_allow_html=True,
    )

    claim_input = st.text_area(
        "Claim or question",
        placeholder="e.g. 'Tenma is the protagonist of Monster' or 'Who is the protagonist of Monster?'",
        height=80,
    )

    # Extractor filter applies to ALL attribution queries:
    # rebel/deepseek graphs must never be queried together.
    query_extractor = st.radio(
        "Graph to query (extractor)",
        options=settings.AVAILABLE_EXTRACTORS,
        index=settings.AVAILABLE_EXTRACTORS.index(settings.ACTIVE_EXTRACTOR)
        if settings.ACTIVE_EXTRACTOR in settings.AVAILABLE_EXTRACTORS else 0,
        horizontal=True,
    )

    verify_btn = st.button(
        "Verify / Answer",
        type="primary",
        disabled=(not claim_input.strip() or neo4j is None),
    )

    if neo4j is None:
        st.warning("Neo4j not connected. Start Neo4j Desktop first.")

    if verify_btn and claim_input.strip() and neo4j:
        from src.ingestion.output_store import save_attribution

        attributor = get_attributor(semantic_threshold, query_extractor)

        with st.spinner("Parsing input and querying graph..."):
            result = attributor.attribute(claim_input.strip())

        mode_label = "Question" if result.is_question else "Claim"
        st.caption(f"Detected mode: **{mode_label}**")

        # -- Show parsed/resolved triple -----------------------------------
        triple_header = "Answer (triple resolved from graph)" if result.is_question else "Parsed Triple"
        st.markdown(f"#### {triple_header}")
        if result.match_type == "parse_error":
            msg = result.source_chunk or "Unable to parse input."
            st.error(msg)
        elif result.match_type == "not_found" and not result.subject and not result.obj:
            st.warning("No matching triple found in the graph.")
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

        # -- Show attribution result ---------------------------------------
        st.markdown("#### Verification Result")

        if result.match_type == "exact":
            st.markdown(
                """<div class="result-exact">
                    <strong>EXACT MATCH</strong><br>
                    Triple found verbatim in the knowledge graph.
                </div>""",
                unsafe_allow_html=True,
            )
        elif result.match_type == "semantic":
            st.markdown(
                f"""<div class="result-semantic">
                    <strong>SEMANTIC MATCH</strong> -- similarity: {result.similarity:.3f}<br>
                    Predicate matched via cosine similarity (threshold: {semantic_threshold}).
                </div>""",
                unsafe_allow_html=True,
            )
        elif result.match_type == "not_found":
            sim_str = f" (best similarity: {result.similarity:.3f})" if result.similarity > 0 else ""
            st.markdown(
                f"""<div class="result-notfound">
                    <strong>NOT FOUND</strong>{sim_str}<br>
                    No matching triple in graph -- claim cannot be attributed.
                </div>""",
                unsafe_allow_html=True,
            )
        elif result.match_type == "parse_error":
            pass  # already shown above

        # -- Source evidence -----------------------------------------------
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
                            <strong>Title:</strong> {result.source_file}<br>
                            <strong>source_id:</strong> {result.source_id or "n/a"}<br>
                            <strong>Extractor:</strong> {query_extractor}<br>
                            <strong>Passage:</strong> #{result.chunk_index}
                        </p>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Save attribution to JSONL
        save_attribution(
            claim=claim_input.strip(),
            subject=result.subject,
            predicate=result.predicate,
            obj=result.obj,
            match_type=result.match_type,
            similarity=result.similarity,
            source_chunk=result.source_chunk or "",
            source_id=result.source_id or "",
            extractor=query_extractor,
            is_question=result.is_question,
        )
