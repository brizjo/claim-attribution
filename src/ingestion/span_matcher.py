"""
Span matcher — ancora una tripla alla frase del testo ORIGINALE che la
supporta (`claim_span` salvato sull'arco).

Deterministico e a costo zero: split a frasi via regex (niente spaCy: i
passaggi ALCE sono ~100 parole e il caricamento del modello non si ripaga)
e scelta della frase con massimo overlap lessicale sui cue forniti
(subject, object, ed eventuale span restituito dall'LLM).

Serve perché:
  * REBEL non restituisce offset, solo (S, P, O);
  * DeepSeek vede il testo coref-risolto, quindi il suo span va comunque
    ri-ancorato sul testo originale.
"""

from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_WORD = re.compile(r"[a-z0-9]+")

# Parole troppo comuni per contare come evidenza di ancoraggio.
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "was", "were", "are", "be", "been", "by", "with", "as", "that",
    "this", "it", "its", "his", "her", "their", "from", "has", "have",
}


def split_sentences(text: str) -> list[str]:
    """Split a frasi; ricade sul testo intero se non trova confini."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "") if p.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def best_span(text: str, *cues: str) -> str:
    """
    Ritorna la frase di `text` con massimo overlap lessicale sui `cues`.

    Ritorna "" se il testo è vuoto; ritorna l'intero testo se nessuna frase
    condivide alcun token con i cue (meglio un contesto largo che uno errato).
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""

    cue_tokens = set()
    for cue in cues:
        cue_tokens |= _tokens(cue)
    if not cue_tokens:
        return text.strip()

    best, best_score = "", 0.0
    for sent in sentences:
        sent_tokens = _tokens(sent)
        if not sent_tokens:
            continue
        overlap = len(cue_tokens & sent_tokens)
        if not overlap:
            continue
        # Normalizza sui cue (copertura) con penalità lieve sulla lunghezza:
        # a parità di copertura vince la frase più corta = span più preciso.
        score = overlap / len(cue_tokens) + overlap / (len(sent_tokens) + 10)
        if score > best_score:
            best, best_score = sent, score

    return best or text.strip()


# ────────────────────────────────────────────────────────────────────
# Ancoraggio verbatim (pipeline ibrida) — lo span DEVE contenere S e O
# ────────────────────────────────────────────────────────────────────

def content_tokens(text: str) -> set[str]:
    """Token non-stopword, lowercase — unità di confronto per l'ancoraggio."""
    return _tokens(text)


def contains_cue(span: str, cue: str) -> bool:
    """
    True se TUTTI i content token di `cue` compaiono in `span` (token-level).

    Tollera ordine e articoli diversi ("Obama Barack" ⊆ "Barack Obama was...")
    ma NON parole assenti dal testo ("President Obama" su uno span che non
    dice "president" → False).  Un cue senza content token (es. il pronome
    "it") non è ancorabile: False.
    """
    cue_tok = _tokens(cue)
    if not cue_tok:
        return False
    return cue_tok <= _tokens(span)


def _windows(sentences: list[str], size: int) -> list[tuple[int, int]]:
    return [(i, i + size) for i in range(0, max(0, len(sentences) - size + 1))]


def anchor_span(
    text: str,
    subject: str,
    obj: str,
    max_window: int = 2,
) -> str | None:
    """
    Frase (o finestra di `max_window` frasi consecutive) di `text` che contiene
    sia `subject` sia `obj` a livello di token.  `None` se non esiste: la
    tripla non è ancorabile e va scartata dai guardrail.
    """
    sentences = split_sentences(text)
    if not sentences:
        return None
    for size in range(1, max_window + 1):
        for start, end in _windows(sentences, size):
            span = " ".join(sentences[start:end])
            if contains_cue(span, subject) and contains_cue(span, obj):
                return span
    return None


def anchor_span_aligned(
    original: str,
    resolved: str,
    subject: str,
    obj: str,
    max_window: int = 2,
) -> tuple[str, str] | None:
    """
    Ritorna `(span_original, span_resolved)` — l'evidenza verbatim sul testo
    ORIGINALE e la stessa finestra sul testo coref-risolto.

    Serve perché il modello vede il testo risolto ("Barack Obama was born...")
    mentre il testo originale dice "He was born...": ancorare S/O direttamente
    sull'originale scarterebbe triple corrette.  Si ancora sul risolto e si
    riporta la finestra sull'originale per indice di frase (la coref sostituisce
    menzioni in-place, quindi il numero di frasi si conserva).  Se i due split
    non hanno lo stesso numero di frasi si ricade sull'ancoraggio diretto
    sull'originale (nessuna evidenza inventata).
    """
    orig_sents = split_sentences(original)
    res_sents = split_sentences(resolved or original)

    if not orig_sents:
        return None

    if len(orig_sents) == len(res_sents):
        for size in range(1, max_window + 1):
            for start, end in _windows(res_sents, size):
                span_res = " ".join(res_sents[start:end])
                if contains_cue(span_res, subject) and contains_cue(span_res, obj):
                    return " ".join(orig_sents[start:end]), span_res

    span_orig = anchor_span(original, subject, obj, max_window=max_window)
    if span_orig is not None:
        return span_orig, span_orig
    return None
