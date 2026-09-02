"""
Guardrail sulle triple — scarta l'output corrotto prima di JSONL/grafo.

Vale per QUALSIASI produttore (REBEL, DeepSeek, ibrido): una tripla è valida
solo se è *verificabile sul testo*.  Regole (la prima che fallisce vince):

  1. `no_predicate`        — predicato vuoto o composto solo di stopword
                             ("is a", "of") → non asserisce nulla.
  2. `empty_subject` / `empty_object`
                           — vuoto o senza content token (es. "it", "the").
  3. `subject_equals_object`
                           — S e O sono la stessa entità → tautologia.
  4. `no_span`             — nessuna frase del passaggio contiene sia S sia O:
                             la tripla non è ancorabile a evidenza verbatim.
  5. `span_not_verbatim`   — lo span non compare nel testo originale (span
                             riscritto/allucinato dall'LLM).
  6. `subject_not_in_span` / `object_not_in_span`
                           — controllo esplicito sul testo coref-risolto.
  7. `object_is_claim` / `subject_is_claim`
                           — S o O copre quasi tutto lo span: l'LLM ha messo
                             l'intera frase nel campo invece di un'entità.

Il match è a livello di token non-stopword (vedi `span_matcher.contains_cue`):
ordine e articoli diversi passano, parole assenti dal testo no.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ingestion.span_matcher import contains_cue, content_tokens

# Frazione di content token dello span coperti da S o O oltre la quale il
# campo non è più un'entità ma la frase intera.
CLAIM_COVERAGE = 0.8
# Sotto questa lunghezza lo span è già corto: la regola 7 non si applica
# (uno span di 3 token può legittimamente essere quasi tutto l'oggetto).
CLAIM_MIN_SPAN_TOKENS = 5

_WS = re.compile(r"\s+")

# Ordine di applicazione = ordine di dichiarazione; usato anche per la
# tabella dei motivi di scarto in UI.
REASONS = (
    "no_predicate",
    "empty_subject",
    "empty_object",
    "subject_equals_object",
    "no_span",
    "span_not_verbatim",
    "subject_not_in_span",
    "object_not_in_span",
    "subject_is_claim",
    "object_is_claim",
)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:      # `if verdict:` legge come "è valida"
        return self.ok


def _norm_ws(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def _coverage(field: str, span: str) -> float:
    span_tok = content_tokens(span)
    if len(span_tok) < CLAIM_MIN_SPAN_TOKENS:
        return 0.0
    return len(content_tokens(field) & span_tok) / len(span_tok)


def check(
    subject: str,
    predicate: str,
    obj: str,
    span_original: str | None,
    span_resolved: str | None = None,
    original_text: str = "",
) -> Verdict:
    """
    Valuta una tripla già ancorata.

    `span_original` è l'evidenza verbatim salvata sull'output; `span_resolved`
    è la stessa finestra sul testo coref-risolto ed è ciò su cui si verifica la
    presenza di S e O (l'originale può dire "He" dove il modello dice il nome).
    """
    span_res = span_resolved or span_original or ""

    if not content_tokens(predicate):
        return Verdict(False, "no_predicate")
    if not content_tokens(subject):
        return Verdict(False, "empty_subject")
    if not content_tokens(obj):
        return Verdict(False, "empty_object")
    if content_tokens(subject) == content_tokens(obj):
        return Verdict(False, "subject_equals_object")

    if not (span_original or "").strip():
        return Verdict(False, "no_span")
    if original_text and _norm_ws(span_original) not in _norm_ws(original_text):
        return Verdict(False, "span_not_verbatim")

    if not contains_cue(span_res, subject):
        return Verdict(False, "subject_not_in_span")
    if not contains_cue(span_res, obj):
        return Verdict(False, "object_not_in_span")

    if _coverage(subject, span_res) >= CLAIM_COVERAGE:
        return Verdict(False, "subject_is_claim")
    if _coverage(obj, span_res) >= CLAIM_COVERAGE:
        return Verdict(False, "object_is_claim")

    return Verdict(True)
