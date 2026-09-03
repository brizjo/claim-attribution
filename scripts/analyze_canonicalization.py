"""
Analisi della fase di canonicalizzazione — legge
`data/outputs/canonicalization.jsonl` (una riga per menzione).

Cosa stampa:
  1. distribuzione per stadio (assoluto e %) — se la maggioranza si chiude agli
     stadi 1-3 la canonicalizzazione e' in gran parte DETERMINISTICA;
  2. merge effettuati (menzioni -> forma canonica), ordinati per numero di
     menzioni unificate: e' la lista da ispezionare a mano per i falsi merge;
  3. nodi con piu' archi + su quanti passaggi distinti — molti archi su molti
     passaggi = candidato falso merge (nodo-calamita);
  4. conteggio nodi prima/dopo — metrica di efficacia.

Ogni riga di log e' un ESTREMO di arco (una tripla ne produce due: soggetto e
oggetto), quindi "archi" al punto 3 conta gli archi incidenti al nodo.

Uso:
    python scripts/analyze_canonicalization.py
    python scripts/analyze_canonicalization.py --sample-id 1234 --top 30
    python scripts/analyze_canonicalization.py --file data/outputs/canonicalization.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.ingestion.entity_canonicalizer import STAGE_LABELS  # noqa: E402

DEFAULT_FILE = Path(settings.OUTPUT_DIR) / "canonicalization.jsonl"


def load(path: Path, sample_id: str = "") -> list[dict]:
    if not path.is_file():
        raise SystemExit(
            f"File non trovato: {path}\n"
            "Esegui prima un'ingestione (extract -> canonicalize -> write)."
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if sample_id and record.get("sample_id") != sample_id:
            continue
        rows.append(record)
    return rows


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def stage_distribution(rows: list[dict]) -> None:
    total = len(rows) or 1
    counts = Counter(int(r.get("stage", 0)) for r in rows)
    print(f"\n=== Distribuzione per stadio ({len(rows)} menzioni) ===")
    deterministic = 0
    for stage in sorted(STAGE_LABELS):
        n = counts.get(stage, 0)
        if stage <= 3:
            deterministic += n
        confidences = [
            float(r.get("confidence", 1.0)) for r in rows
            if int(r.get("stage", 0)) == stage
        ]
        avg = sum(confidences) / len(confidences) if confidences else 0.0
        print(f"  stadio {stage} ({STAGE_LABELS[stage]:<13}) "
              f"{n:>6}  {100.0 * n / total:>5.1f}%   confidenza media {avg:.3f}")
    print(f"  --> deterministico (stadi 1-3): {deterministic} "
          f"({100.0 * deterministic / total:.1f}%)")


def merges(rows: list[dict], top: int) -> None:
    clusters: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        clusters[_norm(r.get("canonical", ""))].add(r.get("mention", ""))
    merged = [
        (canonical, sorted(mentions))
        for canonical, mentions in clusters.items()
        if len({_norm(m) for m in mentions}) > 1
    ]
    merged.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    print(f"\n=== Merge effettuati ({len(merged)} forme canoniche unificano "
          f"piu' di una menzione) ===")
    print("  (ispezione manuale dei falsi merge: partire dall'alto)")
    for canonical, mentions in merged[:top]:
        display = next((r["canonical"] for r in rows
                        if _norm(r.get("canonical", "")) == canonical), canonical)
        print(f"  [{len(mentions):>2}] {display}")
        for mention in mentions:
            stages = {int(r.get("stage", 0)) for r in rows
                      if r.get("mention") == mention
                      and _norm(r.get("canonical", "")) == canonical}
            print(f"        <- {mention}   (stadio {sorted(stages)})")


def node_degree(rows: list[dict], top: int, min_edges: int) -> None:
    edges: Counter = Counter()
    passages: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        key = _norm(r.get("canonical", ""))
        edges[key] += 1
        passages[key].add(r.get("source_id", ""))
    print(f"\n=== Nodi per archi incidenti (>= {min_edges}) ===")
    print("  molti archi su MOLTI passaggi distinti = candidato falso merge")
    print(f"  {'archi':>6} {'passaggi':>9}  nodo")
    shown = 0
    for key, count in edges.most_common():
        if count < min_edges or shown >= top:
            break
        display = next((r["canonical"] for r in rows
                        if _norm(r.get("canonical", "")) == key), key)
        print(f"  {count:>6} {len(passages[key]):>9}  {display}")
        shown += 1


def node_counts(rows: list[dict]) -> None:
    before = len({_norm(r.get("mention", "")) for r in rows})
    after = len({_norm(r.get("canonical", "")) for r in rows})
    linked = len({_norm(r.get("canonical", "")) for r in rows if r.get("external_id")})
    print("\n=== Efficacia ===")
    print(f"  nodi prima (menzioni distinte):  {before}")
    print(f"  nodi dopo  (forme canoniche):    {after}")
    if before:
        print(f"  riduzione:                       "
              f"{before - after} ({100.0 * (before - after) / before:.1f}%)")
    print(f"  forme canoniche con external_id: {linked}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default=str(DEFAULT_FILE),
                        help="JSONL della canonicalizzazione")
    parser.add_argument("--sample-id", default="", help="filtra su una domanda ALCE")
    parser.add_argument("--top", type=int, default=20, help="righe per sezione")
    parser.add_argument("--min-edges", type=int, default=2,
                        help="soglia archi per la sezione nodi")
    args = parser.parse_args()

    rows = load(Path(args.file), args.sample_id)
    if not rows:
        print("Nessuna riga da analizzare (filtro troppo stretto?).")
        return 1

    samples = {r.get("sample_id", "") for r in rows}
    print(f"File: {args.file}")
    print(f"Domande: {len(samples)} | menzioni: {len(rows)}")

    stage_distribution(rows)
    node_counts(rows)
    merges(rows, args.top)
    node_degree(rows, args.top, args.min_edges)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
