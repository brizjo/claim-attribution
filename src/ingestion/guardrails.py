"""
Guardrail sulle triple — scarta l'output corrotto prima di JSONL/grafo.

Una tripla vale solo se è *verificabile sulla frase sorgente*.  La pipeline
lavora frase per frase, quindi la frase è nota: `claim_span` non viene più
chiesto al modello ma assegnato dalla pipeline, e i guardrail verificano la
tripla CONTRO quella frase.

Regole, nell'ordine di applicazione (la prima che fallisce vince):

  1. `no_predicate`          predicato vuoto o solo stopword ("is a", "of").
  2. `unresolved_reference`  S oppure O contiene un riferimento deittico
                             irrisolto: pronome ("he", "it"), o sintagma
                             "that same year" / "that game" / "the subsequent
                             game".  Sono nodi che non identificano nulla
                             fuori dalla frase — e la coref non li risolve.
  3. `generic_node`          S oppure O è un sostantivo comune non ancorato
                             ("game", "teams", "players"): niente entità
                             nominata (NER), niente PROPN, niente numero/data.
                             Sono i nodi-calamita che collezionano archi da
                             fatti diversi: falsi positivi strutturali.
  4. `empty_subject` / `empty_object`
                             vuoto o senza content token.
  5. `subject_equals_object` tautologia.
  6. `span_not_verbatim`     lo span non compare nel passaggio originale.
  7. `entity_not_in_sentence`
                             S oppure O non compare nella frase sorgente
                             (match normalizzato + fuzzy leggero su morfologia
                             e possessivi).  Becca le triple inferite da
                             conoscenza pregressa invece che dal testo.
  8. `subject_is_claim` / `object_is_claim`
                             S o O copre quasi tutta la frase: non è
                             un'entità, è la frase stessa.

Il match lessicale è a livello di content token; il fuzzy tollera plurali,
possessivi e accenti ("Pelé's" ~ "Pele"), non sinonimi.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional

from config import settings
from src.ingestion.span_matcher import content_tokens

logger = logging.getLogger(__name__)

# Frazione di content token della frase coperti da S o O oltre la quale il
# campo non è più un'entità ma la frase intera.
CLAIM_COVERAGE = 0.8
CLAIM_MIN_SENTENCE_TOKENS = 5
# Soglia di somiglianza per il match fuzzy token-token.
FUZZY_RATIO = 0.85
FUZZY_MIN_PREFIX = 4

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")

REASONS = (
    "no_predicate",
    "unresolved_reference",
    "generic_node",
    "empty_subject",
    "empty_object",
    "subject_equals_object",
    "span_not_verbatim",
    "entity_not_in_sentence",
    "subject_is_claim",
    "object_is_claim",
    "duplicate",
)

# ── Riferimenti deittici irrisolti ───────────────────────────────────

_PRONOUNS = {
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "theirs", "this", "that", "these", "those", "we", "us", "our",
    "you", "your", "i", "me", "my", "who", "whom", "which", "there", "here",
    "himself", "herself", "itself", "themselves", "one", "another", "other",
    "others", "someone", "something", "anyone", "anything",
}

# Determinante deittico in testa al sintagma: "that game", "this season".
_DEICTIC_HEAD = re.compile(r"^(that|this|these|those|such)\b", re.IGNORECASE)
# Possessivo in testa: "his team", "their coach".
_POSSESSIVE_HEAD = re.compile(r"^(his|her|its|their|our|your|my)\b", re.IGNORECASE)
# "the same year", "the following game", "the aforementioned club".
_RELATIVE_HEAD = re.compile(
    r"^(the\s+)?(same|following|previous|subsequent|next|last|later|earlier|"
    r"former|latter|aforementioned|above|preceding)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


def _norm_ws(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def _fold(text: str) -> str:
    """Minuscolo senza accenti — "Pelé" e "Pele" devono coincidere."""
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _fuzzy_tokens(text: str) -> set[str]:
    return set(_WORD.findall(_fold(text)))


def _token_matches(token: str, pool: set[str]) -> bool:
    """Token presente nel pool: esatto, per prefisso comune, o quasi-uguale."""
    if token in pool:
        return True
    for candidate in pool:
        if len(token) >= FUZZY_MIN_PREFIX and len(candidate) >= FUZZY_MIN_PREFIX:
            if token.startswith(candidate[:FUZZY_MIN_PREFIX]) or \
                    candidate.startswith(token[:FUZZY_MIN_PREFIX]):
                if SequenceMatcher(None, token, candidate).ratio() >= FUZZY_RATIO:
                    return True
        elif SequenceMatcher(None, token, candidate).ratio() >= FUZZY_RATIO:
            return True
    return False


def in_sentence(entity: str, sentence: str) -> bool:
    """
    True se ogni content token di `entity` compare nella frase (fuzzy leggero).

    Le stopword sono ignorate: "the 44th president of the United States" si
    verifica su {44th, president, united, states}.
    """
    entity_tokens = {t for t in _fuzzy_tokens(entity) if t not in _STOP_FOLDED}
    if not entity_tokens:
        return False
    pool = _fuzzy_tokens(sentence)
    return all(_token_matches(tok, pool) for tok in entity_tokens)


def is_deictic(entity: str) -> bool:
    """Riferimento che non identifica nulla fuori dalla frase."""
    text = _norm_ws(entity)
    if not text:
        return False
    if _fold(text) in _PRONOUNS:
        return True
    return bool(
        _DEICTIC_HEAD.match(text)
        or _POSSESSIVE_HEAD.match(text)
        or _RELATIVE_HEAD.match(text)
    )


# ── Ancoraggio dell'entità: NER / POS sulla frase ────────────────────

_HAS_DIGIT = re.compile(r"\d")


class EntityAnchorer:
    """
    Decide se un'entità è *identificante* nella sua frase.

    Ancorata se: contiene una cifra (anni, punteggi, quantità), oppure si
    sovrappone a una named entity della frase, oppure almeno un suo token è
    PROPN / NUM / parte di una entità NER.  Un sostantivo comune nudo
    ("game", "teams") non lo è.
    """

    def __init__(self, nlp=None, model: str = settings.SPACY_MODEL):
        self._nlp = nlp
        self._model = model
        # spaCy non e' thread-safe: il runner batch chiama i guardrail da piu'
        # thread mentre le chiamate LLM sono in volo.
        self._lock = threading.Lock()

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self._model)
            except OSError as exc:
                raise RuntimeError(
                    f"modello spaCy '{self._model}' non installato: serve per il "
                    f"guardrail generic_node ({exc})"
                ) from exc
        return self._nlp

    @lru_cache(maxsize=4096)
    def _analyse(self, sentence: str) -> tuple[frozenset, frozenset]:
        """`(token ancorati, token di named entity)` della frase, foldati."""
        with self._lock:
            doc = self._get_nlp()(sentence)
        anchored = {
            _fold(tok.text) for tok in doc
            if tok.pos_ in ("PROPN", "NUM") or tok.ent_type_
        }
        ent_tokens = {
            _fold(tok.text) for ent in doc.ents for tok in ent
        }
        return frozenset(anchored), frozenset(ent_tokens)

    def is_anchored(self, entity: str, sentence: str) -> bool:
        text = _norm_ws(entity)
        if not text:
            return False
        if _HAS_DIGIT.search(text):
            return True
        anchored, ent_tokens = self._analyse(sentence)
        tokens = {t for t in _fuzzy_tokens(text) if t not in _STOP_FOLDED}
        if not tokens:
            return False
        return any(tok in anchored or tok in ent_tokens for tok in tokens)


_DEFAULT_ANCHORER: Optional[EntityAnchorer] = None


def default_anchorer() -> EntityAnchorer:
    """Anchorer condiviso — carica spaCy una volta sola per processo."""
    global _DEFAULT_ANCHORER
    if _DEFAULT_ANCHORER is None:
        _DEFAULT_ANCHORER = EntityAnchorer()
    return _DEFAULT_ANCHORER


# Stopword foldate: stessa lista di span_matcher, in forma senza accenti.
from src.ingestion.span_matcher import _STOP as _STOP_RAW  # noqa: E402

_STOP_FOLDED = {_fold(w) for w in _STOP_RAW}


def _coverage(field: str, sentence: str) -> float:
    sentence_tokens = content_tokens(sentence)
    if len(sentence_tokens) < CLAIM_MIN_SENTENCE_TOKENS:
        return 0.0
    return len(content_tokens(field) & sentence_tokens) / len(sentence_tokens)


def check(
    subject: str,
    predicate: str,
    obj: str,
    sentence: str,
    claim_span: str = "",
    original_text: str = "",
    anchorer: Optional[EntityAnchorer] = None,
) -> Verdict:
    """
    Valuta una tripla contro la sua frase sorgente.

    `sentence` è la frase (coref-risolta) data all'estrattore: è lì che S e O
    devono comparire.  `claim_span` è la frase corrispondente sul testo
    ORIGINALE, salvata come evidenza verbatim; se `original_text` è passato si
    verifica che vi compaia davvero.
    """
    anchorer = anchorer or default_anchorer()

    if not content_tokens(predicate):
        return Verdict(False, "no_predicate")

    for field in (subject, obj):
        if is_deictic(field):
            return Verdict(False, "unresolved_reference")

    if not content_tokens(subject):
        return Verdict(False, "empty_subject")
    if not content_tokens(obj):
        return Verdict(False, "empty_object")
    if content_tokens(subject) == content_tokens(obj):
        return Verdict(False, "subject_equals_object")

    if claim_span and original_text and \
            _norm_ws(claim_span) not in _norm_ws(original_text):
        return Verdict(False, "span_not_verbatim")

    if not in_sentence(subject, sentence) or not in_sentence(obj, sentence):
        return Verdict(False, "entity_not_in_sentence")

    for field in (subject, obj):
        if not anchorer.is_anchored(field, sentence):
            return Verdict(False, "generic_node")

    if _coverage(subject, sentence) >= CLAIM_COVERAGE:
        return Verdict(False, "subject_is_claim")
    if _coverage(obj, sentence) >= CLAIM_COVERAGE:
        return Verdict(False, "object_is_claim")

    return Verdict(True)
