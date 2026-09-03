"""
Estrattore ibrido REBEL + DeepSeek — esperimento di confronto, varianti A e D.

Unità di lavoro = LA FRASE.  Il passaggio ALCE viene coref-risolto e diviso in
frasi; REBEL e DeepSeek vedono la stessa frase, e la pipeline — che la frase la
conosce — assegna `claim_span` da sé.  Il modello NON restituisce più lo span:
chiederglielo produceva span vuoti (38 su 153 nel run precedente) e triple
valide scartate per `no_span`.

    Variante A — VOCABOLARIO (1 chiamata LLM per frase)
        DeepSeek riceve la frase + il vocabolario di predicati di REBEL, NON le
        sue triple.  Misura quanto REBEL serva come *schema* invece che come
        estrattore.

    Variante D — CIECA + VALIDAZIONE (1 chiamata per frase + 1 per passaggio)
        1) DeepSeek estrae dalla frase senza vedere nulla di REBEL;
        2) confronto programmatico (chiave (subject, object) normalizzata);
        3) le sole triple REBEL assenti dal set DeepSeek vengono validate
           sì/no in una seconda chiamata, una per passaggio.

Contabilità REBEL: ogni candidato grezzo riceve ESATTAMENTE uno stato
(`confirmed`, `validated`, `rejected_llm`, `rejected_guardrail`), assegnato per
indice e non per lookup su insiemi — è l'unico modo per cui l'invariante
`matched + rejected == prodotte` regga anche quando due candidati collassano
sulla stessa coppia (subject, object).

Nessuna scrittura su Neo4j: questo modulo produce dati e statistiche.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from src.ingestion import guardrails
from src.ingestion.deepseek_extractor import parse_response
from src.ingestion.span_matcher import align_sentences, content_tokens
from src.llm import json_repair

logger = logging.getLogger(__name__)

VARIANT_A = "A"
VARIANT_D = "D"
VARIANTS = (VARIANT_A, VARIANT_D)

VARIANT_LABELS = {
    VARIANT_A: "A — frase + vocabolario REBEL (1 call/frase)",
    VARIANT_D: "D — DeepSeek cieco + validazione delle sole REBEL residue",
}

# Provenienza della tripla finale (campo `origin` nel JSONL).
ORIGIN_DEEPSEEK = "deepseek"                 # solo DeepSeek
ORIGIN_REBEL_CONFIRMED = "rebel_confirmed"   # prodotta anche da REBEL (accordo)
ORIGIN_REBEL_VALIDATED = "rebel_validated"   # solo REBEL, tenuta dal validatore

# Stato di un candidato REBEL grezzo.
STATUS_CONFIRMED = "confirmed"                  # accordo con DeepSeek
STATUS_VALIDATED = "validated"                  # tenuta dal validatore (solo D)
STATUS_REJECTED_LLM = "rejected_llm"            # scartata dall'LLM
STATUS_REJECTED_GUARDRAIL = "rejected_guardrail"  # tenuta dall'LLM, uccisa dai guardrail
MATCHED_STATUSES = (STATUS_CONFIRMED, STATUS_VALIDATED)


# ════════════════════════════════════════════════════════════════════
# Prompt — nessun campo span: lo assegna la pipeline
# ════════════════════════════════════════════════════════════════════

_SHARED_RULES = """Rules:
1. Use ONLY what the sentence states. Never infer, complete or use outside \
knowledge.
2. Subject and object must be named entities, dates or numbers that appear in \
the sentence. Never a pronoun ("he", "it"), never a deictic phrase ("that same \
year", "that game"), never a bare common noun ("game", "teams", "players").
3. The predicate is a short lowercase relation (2-4 words, no articles).
4. Subject and object must be different.
5. If the sentence states no fact, return an empty list."""

VOCAB_SYSTEM_PROMPT = """You are a precise information extraction engine. You \
convert ONE English sentence into RDF-style triples (subject, predicate, \
object) for a knowledge graph.

A closed relation vocabulary is given: it is the vocabulary of a specialised \
extractor. Reuse one of its relations whenever it fits the sentence; when none \
fits, write your own short predicate rather than forcing a wrong one.

""" + _SHARED_RULES + """

Answer with a single JSON object, no markdown, no commentary:
{"triples": [{"subject": "...", "predicate": "...", "object": "..."}]}"""

VOCAB_USER_TEMPLATE = """Title: {title}

Relation vocabulary ({n} relations):
{vocabulary}

Sentence:
\"\"\"
{sentence}
\"\"\"

Extract the triples of this sentence as JSON."""

BLIND_SYSTEM_PROMPT = """You are a precise information extraction engine. You \
convert ONE English sentence into RDF-style triples (subject, predicate, \
object) for a knowledge graph.

""" + _SHARED_RULES + """

Answer with a single JSON object, no markdown, no commentary:
{"triples": [{"subject": "...", "predicate": "...", "object": "..."}]}"""

BLIND_USER_TEMPLATE = """Title: {title}

Sentence:
\"\"\"
{sentence}
\"\"\"

Extract the triples of this sentence as JSON."""

VALIDATE_SYSTEM_PROMPT = """You are a strict validator. Each candidate triple \
below was produced by a weaker extractor from the sentence quoted next to it, \
and was NOT produced by the stronger extractor. Judge every candidate against \
its own sentence alone.

Answer "keep" only when the sentence explicitly states that exact fact and \
both subject and object appear in it. Answer "discard" when the sentence does \
not state it, when the predicate is meaningless or wrong, when subject equals \
object, or when subject or object is a pronoun, a deictic phrase or a bare \
common noun.

Answer with a single JSON object, no markdown, no commentary:
{"verdicts": [{"index": 0, "verdict": "keep", "reason": "..."}]}"""

VALIDATE_USER_TEMPLATE = """Title: {title}

Candidates to judge:
{candidates}

Return one JSON verdict per candidate index."""


def format_vocabulary(vocabulary: list[str], limit: int = 150) -> str:
    return ", ".join(vocabulary[:limit]) if vocabulary else "(empty)"


def build_vocab_messages(sentence: str, title: str, vocabulary: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": VOCAB_SYSTEM_PROMPT},
        {"role": "user", "content": VOCAB_USER_TEMPLATE.format(
            title=title or "(untitled)", sentence=sentence,
            vocabulary=format_vocabulary(vocabulary), n=len(vocabulary),
        )},
    ]


def build_blind_messages(sentence: str, title: str) -> list[dict]:
    return [
        {"role": "system", "content": BLIND_SYSTEM_PROMPT},
        {"role": "user", "content": BLIND_USER_TEMPLATE.format(
            title=title or "(untitled)", sentence=sentence)},
    ]


def build_validate_messages(title: str, candidates: list[dict]) -> list[dict]:
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            '{}. ("{}", "{}", "{}")\n   sentence: "{}"'.format(
                i, c["subject"], c["predicate"], c["obj"], c["sentence"])
        )
    return [
        {"role": "system", "content": VALIDATE_SYSTEM_PROMPT},
        {"role": "user", "content": VALIDATE_USER_TEMPLATE.format(
            title=title or "(untitled)", candidates="\n".join(lines) or "(none)")},
    ]


def parse_verdicts(content: str) -> dict[int, dict]:
    """
    `{index: {"verdict": "keep"|"discard", "reason": str}}`.

    Parse tollerante al JSON rotto (`src/llm/json_repair.py`): senza recupero
    un array non chiuso farebbe risultare rigettate TUTTE le triple REBEL —
    cioè proprio la metrica in esame.
    """
    out: dict[int, dict] = {}
    for position, item in enumerate(json_repair.records(content or "", "verdicts")):
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            idx = position
        out[idx] = {
            "verdict": str(item.get("verdict", "")).strip().lower(),
            "reason": str(item.get("reason", "")).strip(),
        }
    return out


# ════════════════════════════════════════════════════════════════════
# Dati
# ════════════════════════════════════════════════════════════════════

@dataclass
class SentenceUnit:
    """Una frase: input dei modelli e sorgente dello span."""
    index: int
    original: str          # frase del testo ORIGINALE → claim_span
    resolved: str          # frase coref-risolta → input ai modelli
    rebel: list[dict] = field(default_factory=list)


@dataclass
class HybridTriple:
    subject: str
    predicate: str
    obj: str
    origin: str
    sentence_index: int = 0
    claim_span: str = ""       # frase ORIGINALE, verbatim
    sentence: str = ""         # frase coref-risolta su cui si verifica
    rebel_predicate: str = ""  # predicato REBEL corrispondente, se c'è accordo
    ok: bool = True
    reason: str = ""

    @property
    def key(self) -> tuple:
        return (
            frozenset(content_tokens(self.subject)),
            " ".join(sorted(content_tokens(self.predicate))),
            frozenset(content_tokens(self.obj)),
        )


@dataclass
class RebelCandidate:
    """Candidato REBEL con il suo esito — uno stato e uno solo."""
    subject: str
    predicate: str
    obj: str
    sentence_index: int
    status: str = STATUS_REJECTED_LLM
    reason: str = ""

    @property
    def matched(self) -> bool:
        return self.status in MATCHED_STATUSES


@dataclass
class PassageResult:
    source_id: str
    title: str
    chunk_index: int
    original_text: str
    sample_id: str = ""
    resolved_text: str = ""
    units: list[SentenceUnit] = field(default_factory=list)
    deepseek_raw: list[dict] = field(default_factory=list)  # triple LLM pre-guardrail
    survived: list[HybridTriple] = field(default_factory=list)
    discarded: list[HybridTriple] = field(default_factory=list)
    rebel_candidates: list[RebelCandidate] = field(default_factory=list)
    rebel_seconds: float = 0.0
    llm_seconds: float = 0.0
    llm_calls: int = 0
    error: str = ""

    @property
    def produced(self) -> int:
        return len(self.survived) + len(self.discarded)

    @property
    def discard_reasons(self) -> Counter:
        return Counter(t.reason for t in self.discarded)

    @property
    def rebel_raw(self) -> list[dict]:
        return [{"subject": c.subject, "predicate": c.predicate, "obj": c.obj}
                for c in self.rebel_candidates]

    @property
    def rebel_matched(self) -> int:
        return sum(1 for c in self.rebel_candidates if c.matched)

    @property
    def rebel_rejected(self) -> list[RebelCandidate]:
        return [c for c in self.rebel_candidates if not c.matched]

    @property
    def rebel_kept(self) -> int:
        """Triple FINALI nate da un candidato REBEL (<= rebel_matched: due
        candidati con la stessa coppia (S, O) collassano in una tripla sola)."""
        return sum(1 for t in self.survived
                   if t.origin in (ORIGIN_REBEL_CONFIRMED, ORIGIN_REBEL_VALIDATED))


@dataclass
class RunReport:
    variant: str
    sample_id: str = ""
    question: str = ""
    passages: list[PassageResult] = field(default_factory=list)
    vocabulary: list[str] = field(default_factory=list)

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
        return sum(len(p.rebel_candidates) for p in self.passages)

    @property
    def rebel_matched(self) -> int:
        return self._sum("rebel_matched")

    @property
    def rebel_rejected(self) -> int:
        return sum(len(p.rebel_rejected) for p in self.passages)

    @property
    def rebel_kept(self) -> int:
        return self._sum("rebel_kept")

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


def pair_key(subject: str, obj: str) -> tuple:
    """
    Identità di una tripla nel confronto REBEL ↔ DeepSeek.

    Solo (subject, object) a livello di content token: i due estrattori danno
    quasi sempre predicati diversi per lo stesso fatto ("place of birth" vs
    "born in"), e includere il predicato conterebbe come disaccordo quello che
    è accordo.
    """
    return (frozenset(content_tokens(subject)), frozenset(content_tokens(obj)))


# ════════════════════════════════════════════════════════════════════
# Estrattore
# ════════════════════════════════════════════════════════════════════

class HybridExtractor:
    """
    Esegue una variante su un passaggio già coref-risolto.

    Dipendenze iniettate (REBEL, client DeepSeek, splitter, anchorer) perché in
    Streamlit vivono dietro `@st.cache_resource` e nel runner batch sono
    condivise fra i thread.
    """

    def __init__(self, rebel, deepseek_client, splitter=None,
                 variant: str = VARIANT_A, vocabulary: Optional[list[str]] = None,
                 anchorer=None):
        if variant not in VARIANTS:
            raise ValueError(f"Variante sconosciuta: {variant!r} (attese {VARIANTS})")
        self._rebel = rebel
        self._client = deepseek_client
        self._splitter = splitter
        self._anchorer = anchorer
        self.variant = variant
        self.vocabulary = list(vocabulary or [])

    # ── Frasi ───────────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        if self._splitter is None:
            from src.segmentation.sentence_splitter import SentenceSplitter
            self._splitter = SentenceSplitter()
        return self._splitter.split(text) or [text]

    def build_units(self, chunk: dict, resolved_text: str) -> list[SentenceUnit]:
        """Frasi allineate originale↔risolto, senza ancora eseguire REBEL."""
        original = chunk.get("text", "")
        pairs = align_sentences(self._split(original), self._split(resolved_text or original))
        return [SentenceUnit(index=i, original=o, resolved=r)
                for i, (o, r) in enumerate(pairs)]

    # ── REBEL ───────────────────────────────────────────────────────

    def run_rebel(self, chunk: dict, units: list[SentenceUnit]) -> float:
        """Esegue REBEL su ogni frase e popola `unit.rebel`. Ritorna i secondi."""
        t0 = time.time()
        sent_chunks = [{**chunk, "text": u.resolved} for u in units]
        by_sentence: dict[str, list[dict]] = {}
        for triple in self._rebel.extract(sent_chunks):
            by_sentence.setdefault(triple.chunk_text, []).append({
                "subject": triple.subject,
                "predicate": triple.predicate,
                "obj": triple.obj,
            })
        for unit in units:
            seen, kept = set(), []
            for t in by_sentence.get(unit.resolved, []):
                key = (t["subject"].lower(), t["predicate"].lower(), t["obj"].lower())
                if key in seen:
                    continue
                seen.add(key)
                kept.append(t)
            unit.rebel = kept
        return time.time() - t0

    # ── Guardrail ───────────────────────────────────────────────────

    def _finalize(self, candidates: list[HybridTriple], result: PassageResult) -> None:
        """Applica i guardrail e la deduplica; popola survived/discarded."""
        seen: set = set()
        for cand in candidates:
            verdict = guardrails.check(
                cand.subject, cand.predicate, cand.obj,
                sentence=cand.sentence,
                claim_span=cand.claim_span,
                original_text=result.original_text,
                anchorer=self._anchorer,
            )
            if not verdict.ok:
                cand.ok, cand.reason = False, verdict.reason
                result.discarded.append(cand)
                continue
            if cand.key in seen:
                cand.ok, cand.reason = False, "duplicate"
                result.discarded.append(cand)
                continue
            seen.add(cand.key)
            result.survived.append(cand)

    # ── API pubblica ────────────────────────────────────────────────

    def run_passage(self, chunk: dict, resolved_text: str = "",
                    units: Optional[list[SentenceUnit]] = None,
                    rebel_seconds: float = 0.0) -> PassageResult:
        """
        Esegue la variante su un passaggio.

        `units` permette di riusare le frasi e le triple REBEL già calcolate
        (REBEL gira una volta sola e serve entrambe le varianti).
        """
        original = chunk.get("text", "")
        result = PassageResult(
            source_id=str(chunk.get("source_id", "")),
            title=chunk.get("title", chunk.get("source_file", "")),
            chunk_index=chunk.get("chunk_index", 0),
            original_text=original,
            sample_id=str(chunk.get("sample_id", "")),
            resolved_text=resolved_text or original,
        )
        if not original.strip():
            result.error = "empty passage"
            return result

        try:
            if units is None:
                units = self.build_units(chunk, result.resolved_text)
                rebel_seconds = self.run_rebel(chunk, units)
            result.units = units
            result.rebel_seconds = rebel_seconds
            result.rebel_candidates = [
                RebelCandidate(subject=t["subject"], predicate=t["predicate"],
                               obj=t["obj"], sentence_index=u.index)
                for u in units for t in u.rebel
            ]

            candidates = (self._variant_a(result) if self.variant == VARIANT_A
                          else self._variant_d(result))
            self._finalize(candidates, result)
            self._settle_rebel(result)

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error("Hybrid %s failed on source_id=%s: %s",
                         self.variant, result.source_id, exc)
        return result

    # ── Chiamate LLM per frase ──────────────────────────────────────

    def _extract_sentence(self, result: PassageResult,
                          messages: list[dict]) -> list[dict]:
        t0 = time.time()
        content = self._client.chat(messages, json_mode=True)
        result.llm_seconds += time.time() - t0
        result.llm_calls += 1
        return parse_response(content)

    def _to_triples(self, items: list[dict], unit: SentenceUnit) -> list[HybridTriple]:
        """Le triple dell'LLM ereditano lo span dalla frase che le ha generate."""
        return [
            HybridTriple(
                subject=i["subject"], predicate=i["predicate"], obj=i["obj"],
                origin=ORIGIN_DEEPSEEK, sentence_index=unit.index,
                claim_span=unit.original, sentence=unit.resolved,
            )
            for i in items
        ]

    # ── Variante A: frase + vocabolario REBEL ───────────────────────

    def _variant_a(self, result: PassageResult) -> list[HybridTriple]:
        out: list[HybridTriple] = []
        for unit in result.units:
            items = self._extract_sentence(
                result, build_vocab_messages(unit.resolved, result.title,
                                             self.vocabulary))
            result.deepseek_raw.extend(items)
            triples = self._to_triples(items, unit)
            self._mark_agreement(triples, unit)
            out.extend(triples)
        return out

    # ── Variante D: cieca + validazione delle sole REBEL residue ────

    def _variant_d(self, result: PassageResult) -> list[HybridTriple]:
        out: list[HybridTriple] = []
        pending: list[dict] = []   # candidati REBEL non prodotti da DeepSeek

        for unit in result.units:
            items = self._extract_sentence(
                result, build_blind_messages(unit.resolved, result.title))
            result.deepseek_raw.extend(items)
            triples = self._to_triples(items, unit)
            self._mark_agreement(triples, unit)
            out.extend(triples)

            ds_pairs = {pair_key(t.subject, t.obj) for t in triples}
            for cand in result.rebel_candidates:
                if cand.sentence_index != unit.index:
                    continue
                if pair_key(cand.subject, cand.obj) in ds_pairs:
                    continue  # accordo: gestito da _settle_rebel
                pending.append({
                    "subject": cand.subject, "predicate": cand.predicate,
                    "obj": cand.obj, "sentence": unit.resolved, "unit": unit,
                    "candidate": cand,
                })

        if not pending:
            return out

        t0 = time.time()
        content = self._client.chat(
            build_validate_messages(result.title, pending), json_mode=True)
        result.llm_seconds += time.time() - t0
        result.llm_calls += 1
        verdicts = parse_verdicts(content)

        for i, item in enumerate(pending):
            verdict = verdicts.get(i)
            cand: RebelCandidate = item["candidate"]
            unit: SentenceUnit = item["unit"]
            if verdict and verdict["verdict"] == "keep":
                cand.status = STATUS_VALIDATED
                cand.reason = verdict.get("reason", "")
                out.append(HybridTriple(
                    subject=cand.subject, predicate=cand.predicate, obj=cand.obj,
                    origin=ORIGIN_REBEL_VALIDATED, sentence_index=unit.index,
                    claim_span=unit.original, sentence=unit.resolved,
                    rebel_predicate=cand.predicate,
                ))
            else:
                # Nessun verdetto = discard: in dubbio la tripla REBEL non passa.
                cand.status = STATUS_REJECTED_LLM
                cand.reason = ((verdict or {}).get("reason")
                               or "no verdict from validator")
        return out

    # ── Accordo REBEL ↔ DeepSeek ────────────────────────────────────

    @staticmethod
    def _mark_agreement(triples: list[HybridTriple], unit: SentenceUnit) -> None:
        """Marca come `rebel_confirmed` le triple LLM che REBEL ha proposto."""
        rebel_by_pair = {pair_key(t["subject"], t["obj"]): t for t in unit.rebel}
        for triple in triples:
            match = rebel_by_pair.get(pair_key(triple.subject, triple.obj))
            if match:
                triple.origin = ORIGIN_REBEL_CONFIRMED
                triple.rebel_predicate = match["predicate"]

    @staticmethod
    def _settle_rebel(result: PassageResult) -> None:
        """
        Stato finale di ogni candidato REBEL — uno e uno solo.

        Si confronta la coppia (S, O) del candidato con le triple finali della
        STESSA frase: sopravvissuta -> `confirmed` (o `validated`, se e' stato
        il validatore a salvarla in variante D); scartata dai guardrail ->
        `rejected_guardrail`; assente -> `rejected_llm`.

        Anche un candidato gia' marcato `validated` deve ripassare di qui:
        il verdetto dell'LLM non basta se poi i guardrail uccidono la tripla,
        altrimenti verrebbe contato fra le confermate senza essere in output.
        """
        survived_pairs = {(t.sentence_index, pair_key(t.subject, t.obj))
                          for t in result.survived}
        discarded_pairs = {(t.sentence_index, pair_key(t.subject, t.obj)): t.reason
                           for t in result.discarded}

        for cand in result.rebel_candidates:
            key = (cand.sentence_index, pair_key(cand.subject, cand.obj))
            if key in survived_pairs:
                if cand.status != STATUS_VALIDATED:
                    cand.status = STATUS_CONFIRMED
                    cand.reason = ""
            elif key in discarded_pairs:
                cand.status = STATUS_REJECTED_GUARDRAIL
                cand.reason = discarded_pairs[key]
            else:
                cand.status = STATUS_REJECTED_LLM
                if not cand.reason or cand.status == STATUS_VALIDATED:
                    cand.reason = "not produced by DeepSeek"


def rebel_vocabulary(units_or_predicates, limit: int = 150) -> list[str]:
    """
    Vocabolario di predicati REBEL, ordinato per frequenza.

    `rebel-large` è un seq2seq BART: il suo `id2label` è `LABEL_0/1/2`, non
    contiene le relazioni.  Il vocabolario si ricava quindi dall'output di
    REBEL sul corpus in esame — è il vocabolario che REBEL usa davvero.
    """
    counter: Counter = Counter()
    for item in units_or_predicates:
        if isinstance(item, str):
            counter[item] += 1
        else:
            for triple in getattr(item, "rebel", []):
                counter[triple["predicate"]] += 1
    return [pred for pred, _ in counter.most_common(limit) if pred]
