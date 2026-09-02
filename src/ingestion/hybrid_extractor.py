"""
Estrattore ibrido REBEL + DeepSeek — due varianti di debug.

La pipeline non è più "REBEL *oppure* DeepSeek": i due modelli lavorano
insieme.  Poiché la resa di REBEL è bassa, servono numeri per decidere se
tenerlo: entrambe le varianti misurano quante triple REBEL sopravvivono al
giudizio dell'LLM.

    Variante A — CORRECT (1 chiamata LLM per passaggio)
        REBEL sulle frasi → DeepSeek riceve passaggio + triple REBEL e
        produce il set finale: corregge le triple sbagliate, scarta quelle
        non supportate dal testo, aggiunge le mancanti.

    Variante B — 2 PASSATE (2 chiamate LLM per passaggio)
        1) DeepSeek estrae dal testo da solo, senza vedere REBEL;
        2) REBEL estrae; DeepSeek fa da validatore sulle sole triple REBEL
           (keep/discard con motivo).  Finale = passata 1 + REBEL accettate.

In entrambe le varianti ogni tripla finale passa per: ancoraggio verbatim
(`span_matcher.anchor_span_aligned`) → guardrail (`guardrails.check`).  Le
scartate NON spariscono: restano nel report con il motivo, per la tabella di
debug in UI.

Ordine fisso a monte: coref → sentence split → estrazione.
Nessuna scrittura su Neo4j: questo modulo produce solo dati + statistiche.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field

from src.ingestion import guardrails
from src.ingestion.deepseek_extractor import (
    SYSTEM_PROMPT as DS_EXTRACT_SYSTEM,
    USER_PROMPT_TEMPLATE as DS_EXTRACT_USER,
    parse_response,
)
from src.ingestion.span_matcher import anchor_span_aligned, content_tokens
from src.llm import json_repair

logger = logging.getLogger(__name__)

VARIANT_A = "A"
VARIANT_B = "B"
VARIANTS = (VARIANT_A, VARIANT_B)

VARIANT_LABELS = {
    VARIANT_A: "A — correct (REBEL → DeepSeek corregge e completa)",
    VARIANT_B: "B — 2 passate (DeepSeek estrae → DeepSeek valida REBEL)",
}

# Provenienza della tripla finale (campo `origin` nel JSONL).
ORIGIN_DEEPSEEK = "deepseek"                 # prodotta dall'LLM, assente in REBEL
ORIGIN_REBEL_CONFIRMED = "rebel_confirmed"   # proposta da REBEL, tenuta dall'LLM


# ════════════════════════════════════════════════════════════════════
# Prompt
# ════════════════════════════════════════════════════════════════════

_SHARED_RULES = """Rules:
1. Use ONLY facts stated explicitly in the passage. Never infer, complete or \
use outside knowledge.
2. Subject and object must be concrete named entities, dates, numbers or short \
noun phrases that appear in the passage. Never a pronoun, never a whole \
sentence.
3. The predicate is a short lowercase verb phrase or relation name (2-4 words, \
no articles), e.g. "born in", "member of", "won", "has population".
4. For every triple, quote in "claim_span" the shortest VERBATIM sentence of \
the passage supporting it. Copy it character by character.
5. Subject and object must both be contained in that claim_span.
6. Reject anything the passage does not state: a triple with no real predicate, \
with subject equal to object, or with a whole sentence as object."""

CORRECT_SYSTEM_PROMPT = """You are a precise information extraction engine \
working in a hybrid pipeline. A weaker model (REBEL, closed relation \
vocabulary) already produced candidate triples for the passage. Your job is to \
return the FINAL, CORRECT set of (subject, predicate, object) triples:

* fix every REBEL candidate that is wrong, incomplete or badly worded;
* drop every REBEL candidate the passage does not support;
* ADD every fact REBEL missed - the final set must cover the whole passage.

""" + _SHARED_RULES + """

Mark "from_rebel": true when the triple comes from a REBEL candidate (kept or \
corrected), false when you added it yourself.

Answer with a single JSON object, no markdown, no commentary:
{"triples": [{"subject": "...", "predicate": "...", "object": "...", \
"claim_span": "...", "from_rebel": true}]}"""

CORRECT_USER_TEMPLATE = """Title: {title}

Passage:
\"\"\"
{text}
\"\"\"

REBEL candidate triples (may contain errors, may be incomplete):
{candidates}

Return the corrected and completed triple set as JSON."""

VALIDATE_SYSTEM_PROMPT = """You are a strict validator in a hybrid extraction \
pipeline. Triples were already extracted from the passage by a stronger model. \
Now you must judge candidate triples produced by a weaker model (REBEL) \
against the passage ALONE.

For each candidate answer "keep" or "discard".
Discard when: the passage does not state the fact; subject or object is not in \
the passage; the predicate is empty or meaningless; subject equals object; the \
object is a whole sentence; the triple duplicates a fact already extracted.
Keep ONLY candidates that are verbatim-supported and add information.

""" + _SHARED_RULES + """

For a kept candidate also return "claim_span": the verbatim sentence of the \
passage that supports it.

Answer with a single JSON object, no markdown, no commentary:
{"verdicts": [{"index": 0, "verdict": "keep", "claim_span": "...", \
"reason": "..."}]}"""

VALIDATE_USER_TEMPLATE = """Title: {title}

Passage:
\"\"\"
{text}
\"\"\"

Triples already extracted (do not re-judge them, use them only to spot
duplicates):
{existing}

REBEL candidates to judge:
{candidates}

Return one JSON verdict per candidate index."""


def _format_candidates(triples: list[dict]) -> str:
    if not triples:
        return "(none)"
    return "\n".join(
        '{}. ("{}", "{}", "{}")'.format(i, t["subject"], t["predicate"], t["obj"])
        for i, t in enumerate(triples)
    )


def build_correct_messages(text: str, title: str, rebel: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": CORRECT_SYSTEM_PROMPT},
        {"role": "user", "content": CORRECT_USER_TEMPLATE.format(
            title=title or "(untitled)", text=text,
            candidates=_format_candidates(rebel),
        )},
    ]


def build_extract_messages(text: str, title: str) -> list[dict]:
    """Passata 1 della variante B — stesso prompt dell'estrattore DeepSeek."""
    return [
        {"role": "system", "content": DS_EXTRACT_SYSTEM},
        {"role": "user", "content": DS_EXTRACT_USER.format(
            title=title or "(untitled)", text=text)},
    ]


def build_validate_messages(
    text: str, title: str, rebel: list[dict], existing: list[dict],
) -> list[dict]:
    return [
        {"role": "system", "content": VALIDATE_SYSTEM_PROMPT},
        {"role": "user", "content": VALIDATE_USER_TEMPLATE.format(
            title=title or "(untitled)", text=text,
            candidates=_format_candidates(rebel),
            existing=_format_candidates(existing) if existing else "(none)",
        )},
    ]


def parse_verdicts(content: str) -> dict[int, dict]:
    """
    `{index: {"verdict": "keep"|"discard", "reason": str, "claim_span": str}}`.

    Il parse è tollerante al JSON rotto (`src/llm/json_repair.py`): senza
    recupero, un array non chiuso dal validatore farebbe risultare TUTTE le
    triple REBEL come rigettate, falsando la metrica che serve a decidere se
    tenere REBEL.  Un indice mancante resta comunque "discard" per il
    chiamante: in dubbio la tripla REBEL non entra nel set finale.
    """
    out: dict[int, dict] = {}
    for position, item in enumerate(json_repair.records(content or "", "verdicts")):
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            # Verdetto senza indice: si assume l'ordine dei candidati.
            idx = position
        out[idx] = {
            "verdict": str(item.get("verdict", "")).strip().lower(),
            "reason": str(item.get("reason", "")).strip(),
            "claim_span": str(item.get("claim_span", "")).strip(),
        }
    return out


# ════════════════════════════════════════════════════════════════════
# Dati
# ════════════════════════════════════════════════════════════════════

@dataclass
class HybridTriple:
    subject: str
    predicate: str
    obj: str
    origin: str
    claim_span: str = ""       # evidenza VERBATIM sul testo ORIGINALE
    span_resolved: str = ""    # stessa finestra sul testo coref-risolto
    ok: bool = True
    reason: str = ""           # motivo di scarto (guardrail) se ok=False

    @property
    def key(self) -> tuple:
        return (
            frozenset(content_tokens(self.subject)),
            " ".join(sorted(content_tokens(self.predicate))),
            frozenset(content_tokens(self.obj)),
        )


@dataclass
class RejectedRebel:
    """Tripla REBEL non sopravvissuta al giudizio dell'LLM o ai guardrail."""
    subject: str
    predicate: str
    obj: str
    reason: str = ""


@dataclass
class PassageResult:
    source_id: str
    title: str
    chunk_index: int
    original_text: str
    resolved_text: str = ""
    sentences: list[str] = field(default_factory=list)
    rebel_raw: list[dict] = field(default_factory=list)
    deepseek_first_pass: list[dict] = field(default_factory=list)  # solo variante B
    survived: list[HybridTriple] = field(default_factory=list)
    discarded: list[HybridTriple] = field(default_factory=list)
    rebel_rejected: list[RejectedRebel] = field(default_factory=list)
    # Triple REBEL grezze confluite in una tripla finale. NON coincide con
    # `rebel_kept`: due candidati REBEL con lo stesso (subject, object) e
    # predicati diversi collassano in un'unica tripla finale.  Invariante:
    # rebel_matched + len(rebel_rejected) == len(rebel_raw).
    rebel_matched: int = 0
    rebel_seconds: float = 0.0
    llm_seconds: float = 0.0
    llm_calls: int = 0
    error: str = ""

    @property
    def produced(self) -> int:
        """Triple prodotte dalla pipeline, prima dei guardrail."""
        return len(self.survived) + len(self.discarded)

    @property
    def discard_reasons(self) -> Counter:
        return Counter(t.reason for t in self.discarded)

    @property
    def rebel_kept(self) -> int:
        """Triple FINALI nate da un candidato REBEL (contributo netto)."""
        return sum(1 for t in self.survived if t.origin == ORIGIN_REBEL_CONFIRMED)


@dataclass
class RunReport:
    variant: str
    sample_id: str
    question: str
    passages: list[PassageResult] = field(default_factory=list)

    def _sum(self, attr: str) -> int:
        return sum(getattr(p, attr) for p in self.passages)

    @property
    def produced(self) -> int:
        return self._sum("produced")

    @property
    def survived(self) -> int:
        return sum(len(p.survived) for p in self.passages)

    @property
    def rebel_produced(self) -> int:
        return sum(len(p.rebel_raw) for p in self.passages)

    @property
    def rebel_rejected(self) -> int:
        return sum(len(p.rebel_rejected) for p in self.passages)

    @property
    def rebel_kept(self) -> int:
        return self._sum("rebel_kept")

    @property
    def rebel_matched(self) -> int:
        return self._sum("rebel_matched")

    @property
    def discard_reasons(self) -> Counter:
        total: Counter = Counter()
        for p in self.passages:
            total.update(p.discard_reasons)
        return total

    @property
    def llm_calls(self) -> int:
        return self._sum("llm_calls")

    @property
    def seconds(self) -> float:
        return sum(p.rebel_seconds + p.llm_seconds for p in self.passages)

    @property
    def errors(self) -> list[str]:
        return [f"{p.source_id}: {p.error}" for p in self.passages if p.error]


# ════════════════════════════════════════════════════════════════════
# Estrattore
# ════════════════════════════════════════════════════════════════════

class HybridExtractor:
    """
    Orchestra REBEL + DeepSeek su un passaggio già coref-risolto.

    Le dipendenze sono iniettate (REBEL, client DeepSeek, splitter) perché in
    Streamlit vivono dietro `@st.cache_resource`: costruirle qui ricaricherebbe
    i pesi a ogni run.
    """

    def __init__(self, rebel, deepseek_client, splitter=None, variant: str = VARIANT_A):
        if variant not in VARIANTS:
            raise ValueError(f"Variante sconosciuta: {variant!r} (attese {VARIANTS})")
        self._rebel = rebel
        self._client = deepseek_client
        self._splitter = splitter
        self.variant = variant

    # ── Split frasi (lazy: carica spaCy solo al primo uso) ──────────

    def _split(self, text: str) -> list[str]:
        if self._splitter is None:
            from src.segmentation.sentence_splitter import SentenceSplitter
            self._splitter = SentenceSplitter()
        return self._splitter.split(text) or [text]

    # ── REBEL ───────────────────────────────────────────────────────

    def _run_rebel(self, chunk: dict, sentences: list[str]) -> list[dict]:
        sent_chunks = [{**chunk, "text": s} for s in sentences]
        seen, out = set(), []
        for t in self._rebel.extract(sent_chunks):
            key = (t.subject.lower(), t.predicate.lower(), t.obj.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({"subject": t.subject, "predicate": t.predicate, "obj": t.obj})
        return out

    # ── Ancoraggio + guardrail ──────────────────────────────────────

    def _finalize(
        self, candidates: list[HybridTriple], original: str, resolved: str,
    ) -> tuple[list[HybridTriple], list[HybridTriple]]:
        """Ancora ogni tripla al testo originale, applica i guardrail, dedup."""
        survived: list[HybridTriple] = []
        discarded: list[HybridTriple] = []
        seen: set = set()

        for cand in candidates:
            spans = anchor_span_aligned(original, resolved, cand.subject, cand.obj)
            cand.claim_span, cand.span_resolved = spans if spans else ("", "")

            verdict = guardrails.check(
                cand.subject, cand.predicate, cand.obj,
                cand.claim_span, cand.span_resolved, original,
            )
            if not verdict.ok:
                cand.ok, cand.reason = False, verdict.reason
                discarded.append(cand)
                continue

            if cand.key in seen:
                cand.ok, cand.reason = False, "duplicate"
                discarded.append(cand)
                continue

            seen.add(cand.key)
            survived.append(cand)

        return survived, discarded

    # ── API pubblica ────────────────────────────────────────────────

    def run_passage(self, chunk: dict, resolved_text: str = "") -> PassageResult:
        original = chunk.get("text", "")
        result = PassageResult(
            source_id=str(chunk.get("source_id", "")),
            title=chunk.get("title", chunk.get("source_file", "")),
            chunk_index=chunk.get("chunk_index", 0),
            original_text=original,
            resolved_text=resolved_text or original,
        )
        if not original.strip():
            result.error = "empty passage"
            return result

        try:
            result.sentences = self._split(result.resolved_text)
            t0 = time.time()
            result.rebel_raw = self._run_rebel(chunk, result.sentences)
            result.rebel_seconds = time.time() - t0

            candidates = (self._variant_a(result) if self.variant == VARIANT_A
                          else self._variant_b(result))

            result.survived, result.discarded = self._finalize(
                candidates, original, result.resolved_text,
            )
            self._account_rebel(result)

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error("Hybrid %s failed on source_id=%s: %s",
                         self.variant, result.source_id, exc)

        return result

    # ── Variante A: DeepSeek corregge/completa REBEL ────────────────

    def _variant_a(self, result: PassageResult) -> list[HybridTriple]:
        t0 = time.time()
        content = self._client.chat(
            build_correct_messages(result.resolved_text, result.title, result.rebel_raw),
            json_mode=True,
        )
        result.llm_seconds += time.time() - t0
        result.llm_calls += 1

        rebel_keys = {_pair_key(t["subject"], t["obj"]) for t in result.rebel_raw}
        out = []
        for item in parse_response(content):
            # `origin` deterministico: la provenienza NON viene dal flag
            # dell'LLM ma dal confronto (subject, object) con i candidati REBEL.
            origin = (ORIGIN_REBEL_CONFIRMED
                      if _pair_key(item["subject"], item["obj"]) in rebel_keys
                      else ORIGIN_DEEPSEEK)
            out.append(HybridTriple(
                subject=item["subject"], predicate=item["predicate"],
                obj=item["obj"], origin=origin,
            ))
        return out

    # ── Variante B: DeepSeek estrae da solo, poi valida REBEL ───────

    def _variant_b(self, result: PassageResult) -> list[HybridTriple]:
        t0 = time.time()
        content = self._client.chat(
            build_extract_messages(result.resolved_text, result.title),
            json_mode=True,
        )
        result.llm_seconds += time.time() - t0
        result.llm_calls += 1

        first_pass = parse_response(content)
        result.deepseek_first_pass = first_pass
        out = [
            HybridTriple(subject=i["subject"], predicate=i["predicate"],
                         obj=i["obj"], origin=ORIGIN_DEEPSEEK)
            for i in first_pass
        ]

        if not result.rebel_raw:
            return out

        t1 = time.time()
        content = self._client.chat(
            build_validate_messages(
                result.resolved_text, result.title, result.rebel_raw, first_pass),
            json_mode=True,
        )
        result.llm_seconds += time.time() - t1
        result.llm_calls += 1

        verdicts = parse_verdicts(content)
        for i, cand in enumerate(result.rebel_raw):
            v = verdicts.get(i)
            if v and v["verdict"] == "keep":
                out.append(HybridTriple(
                    subject=cand["subject"], predicate=cand["predicate"],
                    obj=cand["obj"], origin=ORIGIN_REBEL_CONFIRMED,
                ))
            else:
                # Nessun verdetto = discard: in dubbio la tripla REBEL non passa.
                result.rebel_rejected.append(RejectedRebel(
                    subject=cand["subject"], predicate=cand["predicate"],
                    obj=cand["obj"],
                    reason=(v or {}).get("reason") or "no verdict from validator",
                ))
        return out

    # ── Conteggio REBEL rigettate ───────────────────────────────────

    @staticmethod
    def _account_rebel(result: PassageResult) -> None:
        """
        Una tripla REBEL è "rigettata" se non arriva viva in fondo: scartata
        dall'LLM (variante A: assente dal set finale; variante B: verdetto
        discard) oppure uccisa dai guardrail.  Il motivo distingue i due casi.
        """
        kept_pairs = {_pair_key(t.subject, t.obj) for t in result.survived}
        already = {_pair_key(r.subject, r.obj) for r in result.rebel_rejected}
        guardrail_reason = {
            _pair_key(t.subject, t.obj): t.reason
            for t in result.discarded if t.origin == ORIGIN_REBEL_CONFIRMED
        }

        for cand in result.rebel_raw:
            pair = _pair_key(cand["subject"], cand["obj"])
            # L'ordine conta: un candidato gia' bocciato dal validatore resta
            # bocciato anche se l'LLM ha prodotto per suo conto una tripla con
            # lo stesso (subject, object) — altrimenti verrebbe contato due volte.
            if pair in already:
                continue
            if pair in kept_pairs:
                result.rebel_matched += 1
                continue
            result.rebel_rejected.append(RejectedRebel(
                subject=cand["subject"], predicate=cand["predicate"],
                obj=cand["obj"],
                reason=guardrail_reason.get(pair, "dropped by DeepSeek"),
            ))

    # ── Run su un'intera domanda ALCE ───────────────────────────────

    def run_entry(self, chunks: list[dict], resolver=None, sample_id: str = "",
                  question: str = "", progress=None) -> RunReport:
        report = RunReport(variant=self.variant, sample_id=sample_id, question=question)
        for chunk in chunks:
            if progress:
                progress(f"variant {self.variant}: {chunk.get('source_id')}")
            original = chunk.get("text", "")
            resolved = resolver.resolve(original) if resolver else original
            report.passages.append(self.run_passage(chunk, resolved))
        return report


def _pair_key(subject: str, obj: str) -> tuple:
    """
    Identità di una tripla ai fini del confronto REBEL ↔ set finale.

    Solo (subject, object) a livello di content token: il predicato può essere
    stato *corretto* dall'LLM e la tripla resta la stessa proposta di REBEL.
    """
    return (frozenset(content_tokens(subject)), frozenset(content_tokens(obj)))
