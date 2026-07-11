"""
Question parser — converts a natural-language question into a partial
triple QuerySpec (subject, predicate, object), with None for unknown slots.

Uses Ollama (Llama-3 / Qwen) via LlamaGenerator.  Strict JSON output prompt
with few-shot examples in Italian + English.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from src.generator.llama_generator import LlamaGenerator


@dataclass
class QuerySpec:
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None

    @property
    def known_count(self) -> int:
        return sum(1 for v in (self.subject, self.predicate, self.object) if v)

    @property
    def is_complete(self) -> bool:
        return self.known_count == 3

    def unknown_field(self) -> Optional[str]:
        """Return the single unknown slot name, or None if not exactly one is unknown."""
        slots = {"subject": self.subject, "predicate": self.predicate, "object": self.object}
        unknowns = [k for k, v in slots.items() if not v]
        return unknowns[0] if len(unknowns) == 1 else None


_PROMPT = """Estrai una tripla parziale (Soggetto, Predicato, Oggetto) dalla domanda dell'utente.
Per ogni componente non noto, usa "?".

Restituisci SOLO un oggetto JSON valido con le chiavi "subject", "predicate", "object".
Niente testo extra, niente spiegazioni, niente markdown.

Esempi:
Domanda: "Chi è il protagonista di Monster?"
Output: {{"subject": "?", "predicate": "protagonista", "object": "Monster"}}

Domanda: "Cosa fa Tenma di lavoro?"
Output: {{"subject": "Tenma", "predicate": "lavoro", "object": "?"}}

Domanda: "Quale anime è stato creato da Naoki Urasawa?"
Output: {{"subject": "?", "predicate": "creato da", "object": "Naoki Urasawa"}}

Domanda: "Tenma è il protagonista di Monster"
Output: {{"subject": "Tenma", "predicate": "protagonista", "object": "Monster"}}

Domanda: "Where was Einstein born?"
Output: {{"subject": "Einstein", "predicate": "born in", "object": "?"}}

Domanda: "{question}"
Output:"""


class QuestionParser:
    """Parses natural-language questions to a partial QuerySpec via Ollama."""

    def __init__(self, generator: Optional[LlamaGenerator] = None):
        self._gen = generator or LlamaGenerator()

    def parse(self, question: str) -> QuerySpec:
        prompt = _PROMPT.format(question=question.strip().replace('"', "'"))
        try:
            raw = self._gen.generate(prompt).strip()
        except Exception:
            return QuerySpec()

        match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        if not match:
            return QuerySpec()

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return QuerySpec()

        return QuerySpec(
            subject=_clean(data.get("subject")),
            predicate=_clean(data.get("predicate")),
            object=_clean(data.get("object")),
        )


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s == "?":
        return None
    return s
