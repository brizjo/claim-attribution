"""
Coreference resolver — fastcoref (FCoref/LingMessCoref).

FCoref runs on CPU/GPU via HuggingFace transformers.
Falls back to original text on error, but logs the reason so 0.0s
silent failures are visible.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CoreferenceResolver:
    """Resolves coreferences using fastcoref (FCoref model)."""

    def __init__(self):
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from fastcoref import FCoref
        self._model = FCoref()

    def resolve(self, text: str) -> str:
        """
        Returns coreference-resolved text.
        Falls back to original text if fastcoref errors, logging the cause.
        """
        try:
            self._load()
        except Exception as exc:
            logger.warning("fastcoref load failed (%s) — skipping coref", exc)
            return text
        try:
            preds = self._model.predict(texts=[text])
            return self._resolve_clusters(preds[0])
        except Exception as exc:
            logger.warning("fastcoref predict failed (%s) — skipping coref", exc)
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
