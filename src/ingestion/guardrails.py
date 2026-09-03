"""
Guardrail sulle triple — scarta l'output corrotto prima di JSONL/grafo.

Una tripla vale solo se è *verificabile sulla frase sorgente*.  La pipeline
lavora frase per frase, quindi la frase è nota: `claim_span` non viene più
chiesto al modello ma assegnato dalla pipeline, e i guardrail verificano la
tripla CONTRO quella frase.

Regole, nell'ordine di applicazione (la prima che fallisce vince).
Tarate il 2026-09-03 sui falsi positivi del primo run reale (v. lessons.md):

  1. `no_predicate`          predicato VUOTO.  Le copule ("was", "has") NON
                             si scartano più: reggono fatti classificatori
                             legittimi (Bican | was | footballer).
  2. `unresolved_reference`  S oppure O contiene un riferimento deittico
                             irrisolto: pronome ("he", "it"), o sintagma
                             "that same year" / "that game" / "the subsequent
                             game".  Sono nodi che non identificano nulla
                             fuori dalla frase — e la coref non li risolve.
  3. `generic_node`          il SOGGETTO è un sostantivo comune non ancorato
                             ("game", "teams", "match"): niente entità
                             nominata (NER), niente PROPN, niente numero/data,
                             e nemmeno nel titolo.  Sono i nodi-calamita che
                             collezionano archi da fatti diversi.  L'OGGETTO
                             generico è ammesso: "played as | striker" con
                             soggetto ancorato è un fatto verificabile.
  4. `empty_subject` / `empty_object`
                             vuoto o senza content token.
  5. `subject_equals_object` tautologia.
  6. `prepositional_object`  l'OGGETTO inizia con una preposizione ("in
                             Ireland", "on 25 September"): la preposizione fa
                             parte del PREDICATO, non del nodo.  Lasciarla
                             passare crea un nodo distinto per ogni forma
                             preposizionale della stessa entità ("in Ireland"
                             non si unifica mai con "Republic of Ireland":
                             nessuno dei 4 stadi di canonicalizzazione lo
                             risolve, cf. `tasks/issues_canonicalizzazione.md`).
  7. `span_not_verbatim`     lo span non compare nel passaggio originale.
  8. `entity_not_in_sentence`
                             S oppure O non compare né nella frase sorgente
                             né nel TITOLO del passaggio (match normalizzato
                             + fuzzy leggero).  Il titolo è contesto lecito:
                             sta nel prompt, e il modello lo usa — giustamente
                             — come nome canonico dell'entità principale.
  9. `conjunction_mention`   S o O coordina DUE entità nominate distinte
                             ("Fianna Fail and Fine Gael"): non è un nodo, sono
                             due.  Va scisso in due triple con lo stesso
                             predicato — lo chiede il prompt, qui si verifica.
                             Un nome proprio che contiene "and" ("Trinidad and
                             Tobago") NON è una coordinazione: lo distingue la
                             NER, che lo riconosce come entità unica.
 10. `subject_is_claim` / `object_is_claim`
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
    "prepositional_object",
    "span_not_verbatim",
    "entity_not_in_sentence",
    "conjunction_mention",
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


# ── Oggetto preposizionale ───────────────────────────────────────────

# La preposizione appartiene al PREDICATO, non al nodo: "in Ireland" e
# "Republic of Ireland" sono la stessa entita' ma restano due nodi, perche'
# lo stadio 2 della canonicalizzazione lavora per contenimento di token e
# `in` non compare nella forma lunga.  Fermarla qui e' piu' economico che
# recuperarla dopo.
_LEADING_PREPOSITION = re.compile(
    r"^(?:in|on|at|of|to|from|by|for|with|as|into|onto|upon|over|under|"
    r"after|before|during|since|until|till|within|without|between|among|"
    r"amongst|through|throughout|across|against|about|around|near|beside|"
    r"behind|beyond|per|via)\b\s+",
    re.IGNORECASE,
)


def has_leading_preposition(entity: str) -> bool:
    """True se la menzione inizia con una preposizione seguita da altro testo."""
    text = _norm_ws(entity)
    if not text:
        return False
    stripped = _LEADING_PREPOSITION.sub("", text).strip()
    # "Of Mice and Men" senza seguito utile non e' un oggetto preposizionale.
    return bool(stripped) and stripped != text


# ── Coordinazione di entita' ─────────────────────────────────────────

# Una coordinazione richiede una congiunzione ESPLICITA (`and` / `&`).
# `or` NON e' incluso: una disgiunzione ("either of the two main parties")
# e' gia' fermata da `generic_node`, e splittarla produrrebbe due fatti che
# il passaggio non afferma.
_CONJUNCTION = re.compile(r"\s+and\s+|\s*&\s*", re.IGNORECASE)
# La virgola separa solo DENTRO una coordinazione gia' accertata ("X, Y and
# Z").  Da sola non basta: "January 9, 2007" e "Bican, a striker" sono una
# data e un'apposizione, non due entita' coordinate — splittarle a vista
# bocciava triple legittime (visto in test).
_COORD_SPLIT = re.compile(r"\s+and\s+|\s*&\s*|,\s+", re.IGNORECASE)


def is_coordination(
    entity: str,
    sentence: str,
    anchorer: "EntityAnchorer",
    title: str = "",
) -> bool:
    """
    True se la menzione coordina DUE o piu' entita' nominate distinte.

    Il discrimine NON e' la stringa ma la NER: "Procter and Gamble" e
    "Bosnia and Herzegovina" contengono `and` e spaCy li riconosce come UNA
    entita' della frase, quindi non sono coordinazioni.  "Fianna Fail and
    Fine Gael" invece produce due entita' separate, entrambe ancorate.

    Secondo segnale, gratuito: il TITOLO del passaggio.  `en_core_web_sm`
    sbaglia su alcuni toponimi congiuntivi ("Trinidad and Tobago" esce come
    due GPE), ma se la menzione compare VERBATIM nel titolo Wikipedia allora
    e' un nome, non una coordinazione.  Limite residuo noto: quando ne' la
    NER ne' il titolo coprono il caso, un nome proprio con `and` viene
    bocciato.  E' la direzione di errore voluta — meno triple ma piu'
    precise — e il round di repair ha comunque una seconda chance.
    """
    text = _norm_ws(entity)
    if not text or not _CONJUNCTION.search(text):
        return False
    parts = [p.strip() for p in _COORD_SPLIT.split(text) if p.strip()]
    if len(parts) < 2:
        return False
    if anchorer.is_single_entity(text, sentence):
        return False
    if title and _fold(text) in _fold(_norm_ws(title)):
        return False
    return sum(1 for p in parts if anchorer.is_anchored(p, sentence)) >= 2


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
    def _analyse(self, sentence: str) -> tuple[frozenset, frozenset, frozenset]:
        """
        `(token ancorati, token di named entity, entita' INTERE)` della frase.

        Le entita' intere servono a `is_single_entity`: "Trinidad and Tobago"
        e' un solo nodo, non una coordinazione, e la sola cosa che lo sa e'
        la NER — a livello di token e' indistinguibile da "Alpha and Beta".
        """
        with self._lock:
            doc = self._get_nlp()(sentence)
        anchored = {
            _fold(tok.text) for tok in doc
            if tok.pos_ in ("PROPN", "NUM") or tok.ent_type_
        }
        ent_tokens = {
            _fold(tok.text) for ent in doc.ents for tok in ent
        }
        ent_spans = {_norm_ws(_fold(ent.text)) for ent in doc.ents}
        return frozenset(anchored), frozenset(ent_tokens), frozenset(ent_spans)

    def is_anchored(self, entity: str, sentence: str) -> bool:
        text = _norm_ws(entity)
        if not text:
            return False
        if _HAS_DIGIT.search(text):
            return True
        anchored, ent_tokens, _ = self._analyse(sentence)
        tokens = {t for t in _fuzzy_tokens(text) if t not in _STOP_FOLDED}
        if not tokens:
            return False
        return any(tok in anchored or tok in ent_tokens for tok in tokens)

    def is_single_entity(self, entity: str, sentence: str) -> bool:
        """La menzione coincide con UNA named entity intera della frase."""
        text = _norm_ws(_fold(entity))
        if not text:
            return False
        _, _, ent_spans = self._analyse(sentence)
        return text in ent_spans


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
    title: str = "",
) -> Verdict:
    """
    Valuta una tripla contro la sua frase sorgente.

    `sentence` è la frase (coref-risolta) data all'estrattore.  `title` è il
    titolo del passaggio: fa parte del prompt, quindi è CONTESTO LECITO — il
    modello risolve correttamente "The iPhone" nel titolo "iPhone (1st
    generation)" e non va punito per questo.  `claim_span` è la frase
    corrispondente sul testo ORIGINALE; se `original_text` è passato si
    verifica che vi compaia davvero.

    Tarature del 2026-09-03 (sui falsi positivi del primo run reale):
      * `no_predicate` solo per predicato VUOTO: le copule ("was", "has")
        reggono fatti classificatori legittimi (Bican | was | footballer) —
        l'asserzione sta nella coppia copula + oggetto descrittivo.
      * `generic_node` solo sul SOGGETTO: è il soggetto non identificante a
        creare i nodi-calamita ("team", "match"); un oggetto descrittivo con
        soggetto ancorato (played as | striker) è un fatto verificabile.
    """
    anchorer = anchorer or default_anchorer()

    if not _norm_ws(predicate):
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

    # Solo sull'OGGETTO: un soggetto che inizia con una preposizione e' un
    # caso diverso (frase mal segmentata) e cade su `generic_node` o
    # `subject_is_claim`, con un motivo piu' informativo per il repair.
    if has_leading_preposition(obj):
        return Verdict(False, "prepositional_object")

    if claim_span and original_text and \
            _norm_ws(claim_span) not in _norm_ws(original_text):
        return Verdict(False, "span_not_verbatim")

    context = f"{sentence} {title}".strip()
    if not in_sentence(subject, context) or not in_sentence(obj, context):
        return Verdict(False, "entity_not_in_sentence")

    # Dopo i check lessicali: la coordinazione costa una analisi spaCy.
    for field in (subject, obj):
        if is_coordination(field, sentence, anchorer, title):
            return Verdict(False, "conjunction_mention")

    if not anchorer.is_anchored(subject, sentence) and \
            not (title and in_sentence(subject, title)):
        return Verdict(False, "generic_node")

    if _coverage(subject, sentence) >= CLAIM_COVERAGE:
        return Verdict(False, "subject_is_claim")
    if _coverage(obj, sentence) >= CLAIM_COVERAGE:
        return Verdict(False, "object_is_claim")

    return Verdict(True)
