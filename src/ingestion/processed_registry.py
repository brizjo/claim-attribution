"""
Registro dei documenti già processati, per estrattore.

Perché esiste: il check di idempotenza su Neo4j
(`MATCH ()-[r:RELATES_TO {source_id, extractor}]->()`) non vede i documenti
che hanno prodotto ZERO triple — non lasciano archi, quindi verrebbero
riestratti a ogni run (e l'estrazione è la parte costosa).

Formato TSV append-only, una riga per (extractor, source_id):

    extractor \t source_id \t n_triples \t iso_timestamp

I documenti a zero triple restano nel file con `n_triples=0`: sono sia il
marcatore anti-riprocesso sia il dato di copertura dell'estrattore
(`zero_triple_ids()`).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from config import settings

_HEADER = "# extractor\tsource_id\tn_triples\tingested_at"


class ProcessedRegistry:
    """Registro file-based dei doc_id processati, distinti per estrattore."""

    def __init__(self, path: str | Path = settings.PROCESSED_REGISTRY_PATH):
        self._path = Path(path)
        self._rows: Optional[dict[tuple[str, str], dict]] = None

    @property
    def path(self) -> Path:
        return self._path

    # ── I/O ─────────────────────────────────────────────────────────

    def _load(self) -> dict[tuple[str, str], dict]:
        if self._rows is not None:
            return self._rows

        rows: dict[tuple[str, str], dict] = {}
        if self._path.is_file():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                extractor, source_id, n_triples = parts[0], parts[1], parts[2]
                try:
                    n = int(n_triples)
                except ValueError:
                    n = 0
                # L'ultima riga vince (append-only con re-ingest volontari).
                rows[(extractor, source_id)] = {
                    "extractor": extractor,
                    "source_id": source_id,
                    "n_triples": n,
                    "ingested_at": parts[3] if len(parts) > 3 else "",
                }
        self._rows = rows
        return rows

    def reload(self) -> None:
        """Scarta la cache in memoria (altre sessioni possono aver scritto)."""
        self._rows = None

    # ── Query ───────────────────────────────────────────────────────

    def is_processed(self, source_id: str, extractor: str) -> bool:
        return (extractor, str(source_id)) in self._load()

    def processed_ids(self, extractor: Optional[str] = None) -> set[str]:
        return {
            sid for (ext, sid) in self._load()
            if extractor is None or ext == extractor
        }

    def zero_triple_ids(self, extractor: Optional[str] = None) -> set[str]:
        """Doc processati che non hanno prodotto NESSUNA tripla — copertura."""
        return {
            row["source_id"] for row in self._load().values()
            if row["n_triples"] == 0
            and (extractor is None or row["extractor"] == extractor)
        }

    def stats(self, extractor: Optional[str] = None) -> dict:
        rows = [
            r for r in self._load().values()
            if extractor is None or r["extractor"] == extractor
        ]
        zero = sum(1 for r in rows if r["n_triples"] == 0)
        return {
            "docs": len(rows),
            "zero_triple_docs": zero,
            "triples": sum(r["n_triples"] for r in rows),
            "coverage": (len(rows) - zero) / len(rows) if rows else 0.0,
        }

    # ── Write ───────────────────────────────────────────────────────

    def mark(self, source_id: str, extractor: str, n_triples: int) -> None:
        """Registra un documento processato (anche con 0 triple)."""
        source_id = str(source_id)
        ts = datetime.now().isoformat(timespec="seconds")

        self._path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._path.is_file()
        with self._path.open("a", encoding="utf-8") as fh:
            if new_file:
                fh.write(_HEADER + "\n")
            fh.write(f"{extractor}\t{source_id}\t{n_triples}\t{ts}\n")

        self._load()[(extractor, source_id)] = {
            "extractor": extractor,
            "source_id": source_id,
            "n_triples": n_triples,
            "ingested_at": ts,
        }
