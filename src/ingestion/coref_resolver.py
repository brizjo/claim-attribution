"""
Coreference resolver — fastcoref (FCoref/LingMessCoref).

**Fallisce rumorosamente.**  In passato un `except Exception: return text`
mascherava sia i bug di API drift sia l'incompatibilità con transformers >=
4.56 (`'FCorefModel' object has no attribute 'all_tied_weights_keys'`): la
pipeline girava con il testo NON risolto e nessuno se ne accorgeva, finché i
dati non mostravano triple con pronomi e riferimenti deittici.  Ora un
fallimento di caricamento o di inferenza solleva `CorefUnavailable`.

Chi vuole esplicitamente girare senza coref passa `strict=False` (o non usa
il resolver): deve essere una scelta, non un degrado silenzioso.

Limite noto — fastcoref risolve le catene di menzioni (pronomi, nomi
ripetuti), NON i riferimenti deittici temporali/generici come "that same
year" o "the subsequent game": quelli restano nel testo e vengono intercettati
a valle dal guardrail `unresolved_reference`.

Pin obbligatorio: `transformers>=4.41,<4.56` (vedi requirements.txt).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CorefUnavailable(RuntimeError):
    """La coreference resolution non è disponibile: la pipeline non deve proseguire."""


class CoreferenceResolver:
    """Resolves coreferences using fastcoref (FCoref model)."""

    def __init__(self, strict: bool = True):
        self._model = None
        self._strict = strict

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from fastcoref import FCoref
            self._model = FCoref()
        except Exception as exc:
            raise CorefUnavailable(
                f"fastcoref non caricabile ({type(exc).__name__}: {exc}). "
                "Verifica il pin `transformers>=4.41,<4.56` in requirements.txt: "
                "da 4.56 FCorefModel non espone `all_tied_weights_keys`."
            ) from exc

    def check(self) -> tuple[bool, str]:
        """Health check esplicito — per il badge in UI e per il runner batch."""
        try:
            self._load()
        except CorefUnavailable as exc:
            return False, str(exc)
        try:
            probe = "Barack Obama was born in Honolulu. He served as president."
            resolved = self.resolve(probe)
        except CorefUnavailable as exc:
            return False, str(exc)
        if resolved == probe:
            return False, "fastcoref caricato ma non risolve il pronome di prova"
        return True, "fastcoref operativo"

    def resolve(self, text: str) -> str:
        """
        Testo con le coreferenze risolte.

        Solleva `CorefUnavailable` se il modello non si carica o l'inferenza
        fallisce, a meno di `strict=False` (in quel caso ritorna il testo
        originale, loggando l'errore).
        """
        try:
            self._load()
            preds = self._model.predict(texts=[text])
            return self._resolve_clusters(preds[0])
        except CorefUnavailable:
            if self._strict:
                raise
            logger.error("coref non disponibile — testo NON risolto (strict=False)")
            return text
        except Exception as exc:
            if self._strict:
                raise CorefUnavailable(
                    f"fastcoref predict fallito ({type(exc).__name__}: {exc})"
                ) from exc
            logger.error("coref predict fallito (%s) — testo NON risolto", exc)
            return text

    @staticmethod
    def _resolve_clusters(result) -> str:
        """
        fastcoref's CorefResult has no built-in text-rewrite method (only
        get_clusters()/get_logit()) — build the resolved text ourselves:
        each cluster's mentions are replaced by that cluster's longest
        mention (proper nouns like "Tenma" beat pronouns like "he").
        Replacements applied right-to-left so char offsets stay valid.
        """
        text = result.text
        replacements: list[tuple[int, int, str]] = []
        for cluster in result.get_clusters(as_strings=False):
            spans = [s for s in cluster if s and None not in s]
            if len(spans) < 2:
                continue
            canonical_start, canonical_end = max(spans, key=lambda s: s[1] - s[0])
            canonical_text = text[canonical_start:canonical_end]
            for start, end in spans:
                if (start, end) == (canonical_start, canonical_end):
                    continue
                replacements.append((start, end, canonical_text))

        for start, end, replacement in sorted(replacements, reverse=True):
            text = text[:start] + replacement + text[end:]
        return text
