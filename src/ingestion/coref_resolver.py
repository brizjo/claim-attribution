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
            return preds[0].get_resolved_text()
        except Exception as exc:
            logger.warning("fastcoref predict failed (%s) — skipping coref", exc)
            return text
