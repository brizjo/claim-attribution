"""
Runner dell'esperimento ibrido — varianti A e D su N domande ASQA.

    python scripts/run_hybrid_experiment.py --questions 10 --variants A D

Sequenza (l'ordine conta: REBEL gira UNA volta sola e serve entrambe le
varianti, e il vocabolario della variante A si ricava dal suo output):

    1. health check coref: se fastcoref non carica, il run si ferma qui.
       Niente run "riuscito" con testo non risolto.
    2. per ogni passaggio: coref -> split in frasi -> allineamento
       originale/risolto -> REBEL su ogni frase.
    3. vocabolario = predicati REBEL osservati, per frequenza.
    4. varianti: chiamate LLM per frase, in parallelo sui passaggi.
    5. salvataggio JSONL + tabelle di analisi (conferma per predicato,
       archi per nodo, differenza A vs D).

Nessuna scrittura su Neo4j: e' un esperimento di confronto.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.ingestion.alce_loader import AlceLoader  # noqa: E402
from src.ingestion.coref_resolver import CoreferenceResolver, CorefUnavailable  # noqa: E402
from src.ingestion.deepseek_extractor import DeepSeekExtractor  # noqa: E402
from src.ingestion import hybrid_analysis as analysis  # noqa: E402
from src.ingestion.guardrails import EntityAnchorer  # noqa: E402
from src.ingestion.hybrid_extractor import (  # noqa: E402
    VARIANTS,
    HybridExtractor,
    RunReport,
    rebel_vocabulary,
)
from src.ingestion.output_store import save_coref, save_hybrid_report  # noqa: E402
from src.ingestion.triple_extractor import TripleExtractor  # noqa: E402
from src.segmentation.sentence_splitter import SentenceSplitter  # noqa: E402

logger = logging.getLogger("hybrid_experiment")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=6,
                    help="thread paralleli sulle chiamate LLM (REBEL resta seriale)")
    ap.add_argument("--vocab-size", type=int, default=150)
    ap.add_argument("--out", default=str(Path(settings.OUTPUT_DIR) / "experiments"))
    ap.add_argument("--no-cache", action="store_true",
                    help="ignora la cache LLM su disco (chiamate fresche)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("fastcoref").setLevel(logging.WARNING)
    logging.getLogger("src.llm.deepseek_client").setLevel(logging.WARNING)

    t_start = time.time()

    # ── 1. Coref obbligatorio ────────────────────────────────────────
    resolver = CoreferenceResolver(strict=True)
    ok, message = resolver.check()
    if not ok:
        logger.error("COREF NON DISPONIBILE: %s", message)
        logger.error("Run interrotto: la pipeline non gira senza coreference.")
        return 2
    logger.info("coref: %s", message)

    loader = AlceLoader()
    if not loader.exists():
        logger.error("corpus ALCE non trovato: %s", loader.path)
        return 2
    entries = loader.entries()[: args.questions]
    logger.info("domande selezionate: %d", len(entries))

    deepseek = DeepSeekExtractor()
    if not deepseek.is_available():
        logger.error("DEEPSEEK_API_KEY non configurata")
        return 2
    client = deepseek.client
    if args.no_cache:
        client.use_cache = False

    splitter = SentenceSplitter()
    anchorer = EntityAnchorer()
    rebel = TripleExtractor()

    # ── 2. Coref + frasi + REBEL (una volta sola) ────────────────────
    prep = HybridExtractor(rebel=rebel, deepseek_client=client, splitter=splitter,
                           variant=VARIANTS[0], anchorer=anchorer)
    prepared = []          # (entry, chunk, units, rebel_seconds)
    n_sentences = 0
    for entry in entries:
        for chunk in entry.docs():
            original = chunk.get("text", "")
            if not original.strip():
                continue
            try:
                resolved = resolver.resolve(original)
            except CorefUnavailable as exc:
                logger.error("coref fallita su %s: %s", chunk.get("source_id"), exc)
                return 2
            save_coref(
                source_id=str(chunk.get("source_id", "")), sample_id=entry.sample_id,
                title=chunk.get("title", ""), chunk_index=chunk.get("chunk_index", 0),
                original_text=original, resolved_text=resolved,
            )
            units = prep.build_units(chunk, resolved)
            seconds = prep.run_rebel(chunk, units)
            n_sentences += len(units)
            prepared.append((entry, {**chunk, "sample_id": entry.sample_id},
                             resolved, units, seconds))
            logger.info("REBEL %s: %d frasi, %d triple, %.1fs",
                        chunk.get("source_id"), len(units),
                        sum(len(u.rebel) for u in units), seconds)

    vocabulary = rebel_vocabulary([u for _, _, _, units, _ in prepared for u in units],
                                  limit=args.vocab_size)
    logger.info("vocabolario REBEL: %d predicati distinti (primi 10: %s)",
                len(vocabulary), vocabulary[:10])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rebel_vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 3. Varianti ──────────────────────────────────────────────────
    reports: dict[str, list[RunReport]] = {}
    for variant in args.variants:
        extractor = HybridExtractor(rebel=rebel, deepseek_client=client,
                                    splitter=splitter, variant=variant,
                                    vocabulary=vocabulary, anchorer=anchorer)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(
                lambda item: extractor.run_passage(
                    item[1], item[2], units=item[3], rebel_seconds=item[4]),
                prepared,
            ))
        by_sample: dict[str, RunReport] = {}
        for (entry, *_), passage in zip(prepared, results):
            report = by_sample.setdefault(entry.sample_id, RunReport(
                variant=variant, sample_id=entry.sample_id,
                question=entry.question, vocabulary=vocabulary))
            report.passages.append(passage)
        reports[variant] = list(by_sample.values())
        for report in reports[variant]:
            save_hybrid_report(report)
        summary = analysis.variant_summary(reports[variant])
        logger.info("variante %s completata in %.1fs — %s", variant, time.time() - t0,
                    json.dumps(summary, ensure_ascii=False))
        for report in reports[variant]:
            for err in report.errors:
                logger.error("variante %s: %s", variant, err)

    # ── 4. Analisi ───────────────────────────────────────────────────
    stats = client.cache_stats()
    logger.info("cache LLM: %d hit, %d miss (hit rate %.0f%%)",
                stats["hits"], stats["misses"], 100 * stats["hit_rate"])

    payload = {
        "questions": [e.sample_id for e in entries],
        "passages": len(prepared),
        "sentences": n_sentences,
        "vocabulary_size": len(vocabulary),
        "llm_cache": stats,
        "seconds": round(time.time() - t_start, 1),
        "variants": {},
    }
    for variant, variant_reports in reports.items():
        payload["variants"][variant] = {
            "summary": analysis.variant_summary(variant_reports),
            "predicate_confirmation": analysis.predicate_confirmation(variant_reports),
            "node_degree": analysis.node_degree(variant_reports, min_edges=2),
            "discard_reasons": analysis.discard_reasons(variant_reports),
        }
    if "A" in reports and "D" in reports:
        merged_a = _merge(reports["A"], "A")
        merged_d = _merge(reports["D"], "D")
        payload["diff_A_D"] = analysis.variant_diff(merged_a, merged_d)

    result_path = out_dir / "hybrid_experiment.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    logger.info("risultati in %s", result_path)
    _print_summary(payload)
    return 0


def _merge(reports: list[RunReport], variant: str) -> RunReport:
    merged = RunReport(variant=variant, sample_id="ALL", question="")
    for report in reports:
        merged.passages.extend(report.passages)
    return merged


def _print_summary(payload: dict) -> None:
    print("\n" + "=" * 72)
    print(f"passaggi {payload['passages']} | frasi {payload['sentences']} | "
          f"vocabolario REBEL {payload['vocabulary_size']} | "
          f"{payload['seconds']}s")
    for variant, data in payload["variants"].items():
        print("-" * 72)
        print(f"VARIANTE {variant}: {json.dumps(data['summary'], ensure_ascii=False)}")
        print("motivi di scarto:", json.dumps(data["discard_reasons"], ensure_ascii=False))
        print("conferma REBEL per predicato (primi 15):")
        for row in data["predicate_confirmation"][:15]:
            print(f"  {row['predicato_rebel']:<28} {row['prodotte']:>4} prodotte "
                  f"{row['confermate']:>4} confermate  {row['%']:>5}%")
        print("nodi con piu' archi (primi 10):")
        for row in data["node_degree"][:10]:
            print(f"  {row['nodo']:<38} archi {row['archi']:>3} "
                  f"passaggi {row['passaggi_distinti']:>3}")
    diff = payload.get("diff_A_D")
    if diff:
        print("-" * 72)
        print(f"A vs D — comuni {diff['comuni']} | solo A {len(diff['solo_A'])} | "
              f"solo D {len(diff['solo_D'])}")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
