"""
ALCE/ASQA loader — unica sorgente del corpus (nessun PDF/TXT).

Il file `asqa_eval_gtr_top100_reranked_oracle.json` è una lista di entry:

    {
      "sample_id": str,
      "question":  str,
      "qa_pairs":  [{"question", "short_answers": [...], ...}],
      "docs":      [{"id", "title", "text", "score",
                     "summary", "extraction", "answers_found": [0/1, ...]}]
    }

Regole:
  * i passaggi SONO già chunk (~100 parole) → nessun chunking
  * si usa SEMPRE `text` originale; `summary`/`extraction` sono generati da un
    LLM a monte e vanno ignorati
  * si usano solo i primi ALCE_DOCS_PER_ENTRY docs (già ri-rankati oracle)
  * `id` del doc è la PROVENIENZA salvata sull'arco (`source_id`)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import settings


@dataclass(frozen=True)
class AlceEntry:
    """Una domanda ASQA con i suoi passaggi top-k."""

    index: int
    sample_id: str
    question: str
    qa_pairs: list[dict]
    _docs: list[dict]

    @property
    def short_answers(self) -> list[list[str]]:
        """Le short answers corrette, una lista per ogni sotto-domanda."""
        return [qa.get("short_answers", []) for qa in self.qa_pairs]

    def docs(self, limit: Optional[int] = None) -> list[dict]:
        """
        Passaggi come chunk dict, nella forma consumata dagli estrattori
        (`text` / `source_file` / `chunk_index`) più i metadati ALCE.

        `source_file` = titolo Wikipedia (label leggibile),
        `source_id`   = doc["id"] (provenienza da salvare sull'arco).
        """
        n = settings.ALCE_DOCS_PER_ENTRY if limit is None else limit
        chunks = []
        for i, doc in enumerate(self._docs[:n]):
            chunks.append({
                "text": doc.get("text", ""),          # SEMPRE l'originale
                "source_file": doc.get("title", ""),
                "source_id": str(doc.get("id", "")),
                "chunk_index": i,
                "title": doc.get("title", ""),
                "sample_id": self.sample_id,
                "answers_found": doc.get("answers_found", []),
            })
        return chunks


class AlceLoader:
    """Carica il JSON ALCE una volta sola e ne espone le entry."""

    def __init__(self, path: str | Path = settings.ALCE_DATA_PATH):
        self._path = Path(path)
        self._entries: Optional[list[AlceEntry]] = None

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def load(self) -> list[AlceEntry]:
        """Carica l'intero file (≈10MB) e lo tiene in memoria."""
        if self._entries is not None:
            return self._entries

        if not self.exists():
            raise FileNotFoundError(
                f"Corpus ALCE non trovato: {self._path}\n"
                "Imposta settings.ALCE_DATA_PATH (o la variabile d'ambiente "
                "ALCE_DATA_PATH) sul file "
                "asqa_eval_gtr_top100_reranked_oracle.json."
            )

        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):          # tollera {"data": [...]}
            raw = raw.get("data", [])

        self._entries = [
            AlceEntry(
                index=i,
                sample_id=str(e.get("sample_id", i)),
                question=e.get("question", ""),
                qa_pairs=e.get("qa_pairs", []),
                _docs=e.get("docs", []),
            )
            for i, e in enumerate(raw)
        ]
        return self._entries

    def entries(self) -> list[AlceEntry]:
        return self.load()

    def get(self, sample_id: str) -> Optional[AlceEntry]:
        return next((e for e in self.load() if e.sample_id == sample_id), None)

    def search(self, query: str, limit: int = 50) -> list[AlceEntry]:
        """Filtro case-insensitive sul testo della domanda."""
        entries = self.load()
        q = query.strip().lower()
        if not q:
            return entries[:limit]
        return [e for e in entries if q in e.question.lower()][:limit]

    def all_source_ids(self, limit: Optional[int] = None) -> list[str]:
        return [d["source_id"] for e in self.load() for d in e.docs(limit)]
