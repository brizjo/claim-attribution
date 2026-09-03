"""
Analisi dei run ibridi — le tabelle che servono a decidere se tenere REBEL.

Tre viste, tutte calcolate dai `RunReport` (nessuna dipendenza da Neo4j):

  * `predicate_confirmation` — tasso di conferma REBEL SCOMPOSTO PER PREDICATO.
    Il tasso aggregato nasconde il fatto interessante: alcune relazioni REBEL
    ("publication date") sono confermate quasi sempre, altre ("participant in",
    "sport") quasi mai.  È a questo livello che si decide se REBEL vada tenuto
    per un sottoinsieme di relazioni invece che buttato del tutto.

  * `node_degree` — archi per nodo sulle triple finali.  Serve a scovare i
    nodi-calamita: entità non identificanti che collezionano archi da fatti
    diversi (il caso "game" con 14 archi da partite diverse).  `passages` e
    `sentences` distinti sono il segnale: molti archi da UNA frase è normale,
    molti archi da passaggi diversi no.

  * `variant_diff` — cosa cattura DeepSeek cieco (D) che la variante con
    vocabolario (A) perde, e viceversa: è il test dell'ancoraggio, cioè quanto
    il vocabolario REBEL distorca l'estrazione.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from src.ingestion.hybrid_extractor import (
    MATCHED_STATUSES,
    ORIGIN_REBEL_CONFIRMED,
    ORIGIN_REBEL_VALIDATED,
    pair_key,
)
from src.ingestion.span_matcher import content_tokens


def _norm(text: str) -> str:
    return " ".join(sorted(content_tokens(text))) or (text or "").strip().lower()


# ── 1. Conferma REBEL per predicato ──────────────────────────────────

def predicate_confirmation(reports) -> list[dict]:
    """
    Righe `predicato_rebel | prodotte | confermate | rigettate | %`.

    "Confermate" = il candidato REBEL è finito in una tripla finale, per
    accordo con DeepSeek (`confirmed`) o per verdetto del validatore
    (`validated`).
    """
    produced: Counter = Counter()
    confirmed: Counter = Counter()
    reasons: dict[str, Counter] = defaultdict(Counter)

    for report in _as_list(reports):
        for passage in report.passages:
            for cand in passage.rebel_candidates:
                pred = cand.predicate.strip().lower() or "(vuoto)"
                produced[pred] += 1
                if cand.status in MATCHED_STATUSES:
                    confirmed[pred] += 1
                else:
                    reasons[pred][cand.status] += 1

    rows = []
    for pred, n in produced.most_common():
        ok = confirmed[pred]
        top_reason = reasons[pred].most_common(1)
        rows.append({
            "predicato_rebel": pred,
            "prodotte": n,
            "confermate": ok,
            "rigettate": n - ok,
            "%": round(100.0 * ok / n, 1) if n else 0.0,
            "motivo_prevalente": top_reason[0][0] if top_reason else "",
        })
    return rows


# ── 2. Archi per nodo ────────────────────────────────────────────────

def node_degree(reports, min_edges: int = 1) -> list[dict]:
    """
    Righe `nodo | archi | come_subject | come_object | passaggi | frasi`.

    Ordinate per archi decrescenti: in cima stanno i candidati nodo-calamita.
    """
    edges: Counter = Counter()
    as_subject: Counter = Counter()
    as_object: Counter = Counter()
    passages: dict[str, set] = defaultdict(set)
    sentences: dict[str, set] = defaultdict(set)
    label: dict[str, str] = {}

    for report in _as_list(reports):
        for passage in report.passages:
            for t in passage.survived:
                for field, counter in ((t.subject, as_subject), (t.obj, as_object)):
                    key = _norm(field)
                    if not key:
                        continue
                    label.setdefault(key, field)
                    edges[key] += 1
                    counter[key] += 1
                    passages[key].add(passage.source_id)
                    sentences[key].add((passage.source_id, t.sentence_index))

    rows = []
    for key, n in edges.most_common():
        if n < min_edges:
            continue
        rows.append({
            "nodo": label.get(key, key),
            "archi": n,
            "come_subject": as_subject[key],
            "come_object": as_object[key],
            "passaggi_distinti": len(passages[key]),
            "frasi_distinte": len(sentences[key]),
        })
    return rows


# ── 3. Differenza fra varianti ───────────────────────────────────────

def _triple_index(report) -> dict:
    """`{(source_id, sentence_index, pair_key): tripla}` sulle sole finali."""
    index = {}
    for passage in report.passages:
        for t in passage.survived:
            index[(passage.source_id, t.sentence_index,
                   pair_key(t.subject, t.obj))] = t
    return index


def variant_diff(report_a, report_d) -> dict:
    """
    Cosa cattura una variante e l'altra no, confrontando le triple finali
    sulla STESSA frase (chiave: source_id + indice frase + coppia S/O).
    """
    index_a = _triple_index(report_a)
    index_d = _triple_index(report_d)

    def _rows(index, keys):
        out = []
        for key in keys:
            t = index[key]
            out.append({
                "source_id": key[0],
                "frase": key[1],
                "subject": t.subject,
                "predicate": t.predicate,
                "object": t.obj,
                "origin": t.origin,
                "claim_span": t.claim_span,
            })
        return out

    only_d = _rows(index_d, [k for k in index_d if k not in index_a])
    only_a = _rows(index_a, [k for k in index_a if k not in index_d])
    common = [k for k in index_a if k in index_d]

    return {
        "solo_D": only_d,
        "solo_A": only_a,
        "comuni": len(common),
        "totale_A": len(index_a),
        "totale_D": len(index_d),
    }


# ── Riepilogo per variante ───────────────────────────────────────────

def variant_summary(reports) -> dict:
    """Totali aggregati su una lista di report della STESSA variante."""
    reports = _as_list(reports)
    if not reports:
        return {}
    rebel_kept_by_origin: Counter = Counter()
    for report in reports:
        for passage in report.passages:
            for t in passage.survived:
                if t.origin in (ORIGIN_REBEL_CONFIRMED, ORIGIN_REBEL_VALIDATED):
                    rebel_kept_by_origin[t.origin] += 1

    produced = sum(r.produced for r in reports)
    survived = sum(r.survived for r in reports)
    rebel_produced = sum(r.rebel_produced for r in reports)
    rebel_matched = sum(r.rebel_matched for r in reports)
    return {
        "variante": reports[0].variant,
        "domande": len(reports),
        "passaggi": sum(len(r.passages) for r in reports),
        "triple prodotte": produced,
        "sopravvissute": survived,
        "scartate": produced - survived,
        "REBEL prodotte": rebel_produced,
        "REBEL confermate": rebel_matched,
        "REBEL rigettate": sum(r.rebel_rejected for r in reports),
        "% conferma REBEL": round(100.0 * rebel_matched / rebel_produced, 1)
        if rebel_produced else 0.0,
        "triple finali da accordo": rebel_kept_by_origin[ORIGIN_REBEL_CONFIRMED],
        "triple finali solo REBEL": rebel_kept_by_origin[ORIGIN_REBEL_VALIDATED],
        "LLM calls": sum(r.llm_calls for r in reports),
        "sec": round(sum(r.seconds for r in reports), 1),
    }


def discard_reasons(reports) -> list[dict]:
    total: Counter = Counter()
    for report in _as_list(reports):
        total.update(report.discard_reasons)
    return [{"motivo": k, "triple": v} for k, v in total.most_common()]


def _as_list(reports) -> list:
    if reports is None:
        return []
    if isinstance(reports, (list, tuple)):
        return list(reports)
    return [reports]
