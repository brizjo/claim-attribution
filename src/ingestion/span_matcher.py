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
