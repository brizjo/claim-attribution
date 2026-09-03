"""
Esperimenti — pipeline ibrida REBEL + DeepSeek (varianti A / D).

Sono STRUMENTI DI SVILUPPO, non funzionalita' del sistema: il tab compare solo
con `SHOW_EXPERIMENTS=1` nell'ambiente (vedi `settings.SHOW_EXPERIMENTS` e
`.env.example`).  Senza la variabile l'app ha i soli tab Corpus ALCE e Claim
Attribution.

Nessuna scrittura su Neo4j: l'output va su `data/outputs/*.jsonl`.
Per un confronto statisticamente utile usare il runner batch
`scripts/run_hybrid_experiment.py`.
"""

from __future__ import annotations

import streamlit as st

from src.ui.resources import (
    get_alce_loader,
    get_debug_coref_resolver,
    get_deepseek,
)


# ── risorse usate solo dagli esperimenti ──────────────────────────────

@st.cache_resource
def get_debug_rebel_extractor():
    """
    REBEL standalone — nessuna dipendenza da Neo4j.

    REBEL e' fuori dalla pipeline principale (2026-09-03): l'ingestione usa
    solo DeepSeek.  Vive qui perche' gli esperimenti esistono proprio per
    misurare quanto REBEL aggiunge (varianti A/D).
    """
    from src.ingestion.triple_extractor import TripleExtractor
    return TripleExtractor()


@st.cache_resource
def get_sentence_splitter():
    """spaCy sentence splitter — usato dalla pipeline ibrida (REBEL per frase)."""
    from src.segmentation.sentence_splitter import SentenceSplitter
    return SentenceSplitter()


@st.cache_resource
def get_entity_anchorer():
    """spaCy NER/POS per il guardrail generic_node — caricato una volta."""
    from src.ingestion.guardrails import EntityAnchorer
    return EntityAnchorer()


def build_hybrid_extractor(variant: str, vocabulary: list[str] | None = None):
    """
    HybridExtractor per variante. NON e' cached: il vocabolario REBEL cambia a
    ogni run; i pesi (REBEL, spaCy, client) arrivano comunque dai cache_resource.
    """
    from src.ingestion.hybrid_extractor import HybridExtractor
    return HybridExtractor(
        rebel=get_debug_rebel_extractor(),
        deepseek_client=get_deepseek().client,
        splitter=get_sentence_splitter(),
        variant=variant,
        vocabulary=vocabulary or [],
        anchorer=get_entity_anchorer(),
    )


def _triple_line(subject: str, predicate: str, obj: str, extra: str = "") -> None:
    st.markdown(
        f'<span class="triple-tag">S: {subject}</span> '
        f'<span class="triple-tag">P: {predicate}</span> '
        f'<span class="triple-tag">O: {obj}</span>'
        + (f' <span class="span-label">{extra}</span>' if extra else ""),
        unsafe_allow_html=True,
    )


def _passage_rows(report) -> list[dict]:
    """Una riga per passaggio + totali. Invariante: confermate + rigettate = REBEL."""
    rows = []
    for p in report.passages:
        rows.append({
            "passaggio": f"[{p.chunk_index}] {p.source_id}",
            "frasi": len(p.units),
            "triple prodotte": p.produced,
            "sopravvissute": len(p.survived),
            "scartate": len(p.discarded),
            "REBEL prodotte": len(p.rebel_candidates),
            "REBEL confermate": p.rebel_matched,
            "REBEL rigettate": len(p.rebel_rejected),
            "triple finali da REBEL": p.rebel_kept,
            "LLM calls": p.llm_calls,
            "sec": round(p.rebel_seconds + p.llm_seconds, 2),
        })
    rows.append({
        "passaggio": "TOTALE",
        "frasi": sum(len(p.units) for p in report.passages),
        "triple prodotte": report.produced,
        "sopravvissute": report.survived,
        "scartate": report.produced - report.survived,
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
    from src.ingestion import hybrid_analysis as analysis
    from src.ingestion.hybrid_extractor import VARIANT_LABELS

    st.markdown(f"#### Variante {VARIANT_LABELS[report.variant]}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Triple prodotte", report.produced)
    c2.metric("Sopravvissute ai guardrail", report.survived)
    c3.metric("REBEL confermate", f"{report.rebel_matched} / {report.rebel_produced}")
    c4.metric("Triple finali da REBEL", report.rebel_kept,
              help="Piu' candidati REBEL con la stessa coppia (S, O) collassano "
                   "in una tripla sola: e' <= 'REBEL confermate'.")

    st.dataframe(pd.DataFrame(_passage_rows(report)),
                 use_container_width=True, hide_index=True)

    col_r, col_p = st.columns(2)
    with col_r:
        reasons = analysis.discard_reasons(report)
        st.caption("Motivi di scarto (guardrail)")
        st.dataframe(pd.DataFrame(reasons) if reasons else pd.DataFrame(
            [{"motivo": "-", "triple": 0}]),
            use_container_width=True, hide_index=True)
    with col_p:
        st.caption("Conferma REBEL per predicato")
        preds = analysis.predicate_confirmation(report)
        st.dataframe(pd.DataFrame(preds) if preds else pd.DataFrame(
            [{"predicato_rebel": "-", "prodotte": 0, "confermate": 0, "%": 0.0}]),
            use_container_width=True, hide_index=True)

    nodes = analysis.node_degree(report, min_edges=2)
    st.caption("Archi per nodo (>=2 archi) — i primi sono i candidati nodo-calamita")
    st.dataframe(pd.DataFrame(nodes) if nodes else pd.DataFrame(
        [{"nodo": "-", "archi": 0, "passaggi_distinti": 0}]),
        use_container_width=True, hide_index=True)

    for err in report.errors:
        st.error(err)

    for p in report.passages:
        with st.expander(
            f"[{p.chunk_index}] {p.title} | `{p.source_id}` | "
            f"{len(p.units)} frasi | {p.produced} prodotte -> "
            f"{len(p.survived)} sopravvissute | REBEL {len(p.rebel_candidates)} -> "
            f"confermate {p.rebel_matched}",
            expanded=False,
        ):
            st.markdown("**Testo originale (evidenza verbatim)**")
            st.markdown(f'<div class="chunk-box">{p.original_text}</div>',
                        unsafe_allow_html=True)
            if p.resolved_text != p.original_text:
                st.markdown("**Testo coref-risolto (input ai modelli)**")
                st.markdown(f'<div class="chunk-box">{p.resolved_text}</div>',
                            unsafe_allow_html=True)
            st.caption(f"REBEL {p.rebel_seconds:.2f}s | DeepSeek {p.llm_calls} call "
                       f"({p.llm_seconds:.2f}s)")

            for unit in p.units:
                st.markdown(f"**Frase {unit.index}** — `{unit.resolved}`")
                if unit.rebel:
                    for t in unit.rebel:
                        _triple_line(t["subject"], t["predicate"], t["obj"], "REBEL")
                else:
                    st.caption("-- nessuna tripla REBEL su questa frase")
                for t in p.survived:
                    if t.sentence_index == unit.index:
                        _triple_line(t.subject, t.predicate, t.obj,
                                     f"FINALE · origin: {t.origin}")
                for t in p.discarded:
                    if t.sentence_index == unit.index:
                        _triple_line(t.subject, t.predicate, t.obj,
                                     f"SCARTATA: {t.reason}")
                st.markdown(f'<div class="span-label">claim_span: {unit.original}</div>',
                            unsafe_allow_html=True)

            rejected = p.rebel_rejected
            if rejected:
                st.markdown(f"**Candidati REBEL rigettati ({len(rejected)})**")
                for c in rejected:
                    _triple_line(c.subject, c.predicate, c.obj,
                                 f"{c.status}: {c.reason}")


def _render_hybrid_tab() -> None:
    """Esperimento di confronto: nessuna scrittura su Neo4j, solo JSONL + UI."""
    import pandas as pd
    from src.ingestion import hybrid_analysis as analysis
    from src.ingestion.coref_resolver import CorefUnavailable
    from src.ingestion.hybrid_extractor import (
        RunReport, VARIANTS, VARIANT_LABELS, rebel_vocabulary,
    )
    from src.ingestion.output_store import save_coref, save_hybrid_report

    st.markdown("### Esperimento ibrido REBEL + DeepSeek")
    st.markdown(
        "<p style='color:#94a3b8;font-size:.9rem;'>"
        "Unita' di lavoro = <strong>la frase</strong>: coref sul passaggio, split in "
        "frasi, poi REBEL e DeepSeek vedono la stessa frase. Lo <code>claim_span</code> "
        "lo assegna la pipeline (frase originale allineata), non il modello."
        "<br><strong>A</strong> - DeepSeek riceve frase + <em>vocabolario</em> dei "
        "predicati REBEL (non le triple). 1 chiamata per frase."
        "<br><strong>D</strong> - DeepSeek estrae dalla frase alla cieca; confronto "
        "programmatico; le sole triple REBEL assenti dal suo set vengono validate "
        "si'/no in una seconda chiamata."
        "<br>Guardrail: <code>entity_not_in_sentence</code>, <code>generic_node</code>, "
        "<code>unresolved_reference</code>, <code>subject_equals_object</code>, "
        "<code>no_predicate</code>."
        "</p>",
        unsafe_allow_html=True,
    )

    loader = get_alce_loader()
    if not loader.exists():
        st.error(f"Corpus non trovato: `{loader.path}`")
        return
    if not get_deepseek().is_available():
        st.warning("`DEEPSEEK_API_KEY` non configurata: l'esperimento non puo' girare.")

    col_q, col_v = st.columns([2, 1])
    with col_q:
        search_q = st.text_input("Filtra domande", key="hyb_search",
                                 placeholder="es. 'world cup', 'president'...")
    with col_v:
        variants = st.multiselect(
            "Varianti", options=list(VARIANTS), default=list(VARIANTS),
            format_func=lambda v: VARIANT_LABELS[v], key="hyb_variants")

    filtered = loader.search(search_q, limit=200)
    if not filtered:
        st.info("Nessuna domanda corrisponde al filtro.")
        return

    selected = st.selectbox("Domanda", options=filtered,
                            format_func=lambda e: e.question, key="hyb_question")
    st.caption(f"sample_id: `{selected.sample_id}` -- {len(selected.docs())} passaggi")

    col_n, col_r = st.columns([1, 3])
    with col_n:
        n_docs = st.number_input("Passaggi", min_value=1, max_value=len(selected.docs()),
                                 value=len(selected.docs()), key="hyb_n_docs")
    with col_r:
        run_btn = st.button("Esegui esperimento", type="primary",
                            use_container_width=True,
                            disabled=(not variants or not get_deepseek().is_available()))
    st.caption(
        "Per un confronto statisticamente utile usa il runner batch: "
        "`python scripts/run_hybrid_experiment.py --questions 10 --variants A D`"
    )

    if run_btn:
        chunks = selected.docs()[:int(n_docs)]
        resolver = get_debug_coref_resolver()
        reports = {}

        with st.status("Esperimento in corso...", expanded=True) as stage:
            # 1. coref + frasi + REBEL: una volta sola, condiviso fra le varianti.
            prep = build_hybrid_extractor(variants[0])
            prepared = []
            try:
                for chunk in chunks:
                    sid = chunk["source_id"]
                    stage.update(label=f"coref + REBEL su {sid}...")
                    resolved = resolver.resolve(chunk.get("text", ""))
                    save_coref(
                        source_id=sid, sample_id=selected.sample_id,
                        title=chunk.get("title", ""),
                        chunk_index=chunk.get("chunk_index", 0),
                        original_text=chunk.get("text", ""), resolved_text=resolved,
                    )
                    units = prep.build_units(chunk, resolved)
                    seconds = prep.run_rebel(chunk, units)
                    prepared.append((chunk, resolved, units, seconds))
            except CorefUnavailable as exc:
                stage.update(label="coref non disponibile", state="error")
                st.error(f"Coreference resolution non disponibile: {exc}")
                return

            vocabulary = rebel_vocabulary(
                [u for _, _, units, _ in prepared for u in units])
            stage.update(label=f"vocabolario REBEL: {len(vocabulary)} predicati")

            # 2. varianti sulle stesse frasi.
            for variant in variants:
                extractor = build_hybrid_extractor(variant, vocabulary)
                report = RunReport(variant=variant, sample_id=selected.sample_id,
                                   question=selected.question, vocabulary=vocabulary)
                for chunk, resolved, units, seconds in prepared:
                    stage.update(label=f"variante {variant}: {chunk['source_id']}...")
                    report.passages.append(extractor.run_passage(
                        chunk, resolved, units=units, rebel_seconds=seconds))
                counts = save_hybrid_report(report)
                reports[variant] = report
                stage.update(label=f"variante {variant}: {counts['survived']} triple salvate")

            stage.update(label="Esperimento completato", state="complete")

        st.session_state["hybrid_reports"] = {
            "sample_id": selected.sample_id, "reports": reports}

    data = st.session_state.get("hybrid_reports")
    if not data or data["sample_id"] != selected.sample_id:
        st.caption("Nessun risultato per questa domanda. Esegui l'esperimento.")
        return

    reports = data["reports"]

    cache = get_deepseek().client.cache_stats()
    st.caption(
        f"Cache LLM: {cache['hits']} hit / {cache['misses']} miss "
        f"(hit rate {cache['hit_rate']:.0%}, attiva: {cache['enabled']}) — "
        "stesso prompt = stessa risposta, i run sono ripetibili."
    )

    if len(reports) > 1:
        st.markdown("#### Confronto varianti")
        st.dataframe(pd.DataFrame([analysis.variant_summary(r) for r in reports.values()]),
                     use_container_width=True, hide_index=True)

    if "A" in reports and "D" in reports:
        diff = analysis.variant_diff(reports["A"], reports["D"])
        st.markdown("#### Cosa cattura D (DeepSeek cieco) che A perde")
        st.caption(f"comuni {diff['comuni']} | solo A {len(diff['solo_A'])} | "
                   f"solo D {len(diff['solo_D'])} — test dell'ancoraggio: quanto il "
                   "vocabolario REBEL distorce l'estrazione")
        col_d, col_a = st.columns(2)
        with col_d:
            st.caption("Solo in D")
            st.dataframe(pd.DataFrame(diff["solo_D"]) if diff["solo_D"] else
                         pd.DataFrame([{"-": "nessuna"}]),
                         use_container_width=True, hide_index=True)
        with col_a:
            st.caption("Solo in A")
            st.dataframe(pd.DataFrame(diff["solo_A"]) if diff["solo_A"] else
                         pd.DataFrame([{"-": "nessuna"}]),
                         use_container_width=True, hide_index=True)

    for report in reports.values():
        _render_report(report)

    st.caption(
        "Output su `data/outputs/`: `triples_hybrid.jsonl` (sopravvissute), "
        "`triples_hybrid_discarded.jsonl` (scartate + motivo), "
        "`hybrid_runs.jsonl` (statistiche). Nessuna scrittura su Neo4j."
    )


def render() -> None:
    """Corpo del tab esperimenti."""
    _render_hybrid_tab()
