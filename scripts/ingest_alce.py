"""
Ingestione batch ALCE -> Neo4j — pipeline template (2026-09-03).

Per ogni domanda ASQA selezionata esegue la pipeline a tre fasi:

    extract_entry   coref -> frasi -> DeepSeek -> guardrail (+ repair round)
    canonicalize    nodi unificati per scope + predicate_embedding (1 batch)
    write_entry     MERGE su Neo4j, chiave arco (predicate, source_id, extractor)

Health check PRIMA di toccare il corpus (fail-fast, exit code dedicato):
    2  fastcoref non carica / non risolve (pin transformers>=4.41,<4.56)
    3  DeepSeek API non raggiungibile o chiave mancante (.env)
    4  Neo4j non raggiungibile (istanza Desktop "RAG", bolt://localhost:7687)

Uso:
    python scripts/ingest_alce.py --limit 10          # prime 10 domande
    python scripts/ingest_alce.py --sample-id ID ...  # domande specifiche
    python scripts/ingest_alce.py --limit 5 --dry-run # niente Neo4j
    python scripts/ingest_alce.py --limit 5 --force   # cancella e riscrive

Idempotente di default: un passaggio gia' processato (registry O grafo) viene
saltato; `--force` cancella gli archi del passaggio e riestrae.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.ingestion.alce_ingestor import AlceIngestor, build_extractor  # noqa: E402
from src.ingestion.alce_loader import AlceLoader  # noqa: E402
from src.ingestion.coref_resolver import CoreferenceResolver  # noqa: E402
from src.ingestion.entity_canonicalizer import EntityCanonicalizer  # noqa: E402
from src.ingestion.output_store import save_ingest_report  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_alce")

EXIT_COREF = 2
EXIT_DEEPSEEK = 3
EXIT_NEO4J = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=0,
                   help="prime N domande del corpus (0 = serve --sample-id)")
    p.add_argument("--sample-id", action="append", default=[],
                   help="sample_id specifico (ripetibile)")
    p.add_argument("--force", action="store_true",
                   help="cancella gli archi esistenti dei passaggi e riestrae")
    p.add_argument("--dry-run", action="store_true",
                   help="extract + canonicalize, NESSUNA scrittura Neo4j")
    p.add_argument("--scope", choices=settings.CANONICALIZATION_SCOPES,
                   default=settings.CANONICALIZATION_SCOPE,
                   help="scope della canonicalizzazione (default: %(default)s)")
    p.add_argument("--no-coref", action="store_true",
                   help="salta la coreference (SCONSIGLIATO: solo debug)")
    return p.parse_args()


def health_checks(resolver: CoreferenceResolver, extractor, client) -> None:
    """Fail-fast: nessuna chiamata costosa se un componente e' rotto."""
    ok, msg = resolver.check()
    if not ok:
        logger.error("Coref health check FALLITO: %s", msg)
        sys.exit(EXIT_COREF)
    logger.info("Coref: %s", msg)

    ok, msg = extractor.check_connection()
    if not ok:
        logger.error("DeepSeek health check FALLITO: %s", msg)
        sys.exit(EXIT_DEEPSEEK)
    logger.info("DeepSeek: %s (model=%s)", msg, extractor.model)

    if client is None:
        logger.warning("DRY RUN: Neo4j non toccato")
        return
    if not client.is_connected():
        logger.error(
            "Neo4j NON raggiungibile su %s (db=%s). Avvia l'istanza 'RAG' "
            "in Neo4j Desktop e riprova.",
            settings.NEO4J_URI, settings.NEO4J_DATABASE,
        )
        sys.exit(EXIT_NEO4J)
    info = client.server_info()
    logger.info("Neo4j: connesso a %s, database=%s", info["uri"], info["database"])


def main() -> None:
    args = parse_args()
    if not args.limit and not args.sample_id:
        logger.error("Serve --limit N oppure --sample-id ID")
        sys.exit(1)

    loader = AlceLoader()
    if not loader.exists():
        logger.error("Corpus ALCE non trovato: %s", settings.ALCE_DATA_PATH)
        sys.exit(1)

    entries = loader.entries()
    if args.sample_id:
        wanted = set(args.sample_id)
        entries = [e for e in entries if e.sample_id in wanted]
        missing = wanted - {e.sample_id for e in entries}
        if missing:
            logger.error("sample_id non trovati nel corpus: %s", sorted(missing))
            sys.exit(1)
    else:
        entries = entries[: args.limit]

    client = None
    if not args.dry_run:
        from src.graph.neo4j_client import Neo4jClient
        client = Neo4jClient()

    resolver = CoreferenceResolver()  # strict: il check sotto ferma il run
    extractor = build_extractor(settings.ACTIVE_EXTRACTOR)
    health_checks(resolver, extractor, client)

    ingestor = AlceIngestor(
        client=client,
        extractor=extractor,
        resolver=resolver,
        use_coref=not args.no_coref,
        canonicalizer=EntityCanonicalizer(scope=args.scope),
    )

    totals: Counter = Counter()
    discard_reasons: Counter = Counter()
    t_start = time.time()

    for i, entry in enumerate(entries, 1):
        logger.info("── [%d/%d] %s — %s", i, len(entries), entry.sample_id,
                    entry.question[:80])
        t0 = time.time()

        report = ingestor.extract_entry(entry, skip_existing=not args.force)
        ingestor.canonicalize_entry(report)
        if not args.dry_run:
            ingestor.write_entry(report, force=args.force)

        errors = [f"{d.source_id}: {d.error}" for d in report.docs if d.error]
        coref_failures = [d.source_id for d in report.docs if d.coref_failed]
        for doc in report.docs:
            for rec in doc.discarded:
                discard_reasons[rec["discard_reason"]] += 1

        totals.update({
            "questions": 1,
            "docs": len(report.processed),
            "skipped": sum(1 for d in report.docs if d.skipped),
            "extracted": report.total_extracted,
            "written": report.total_triples,
            "discarded": report.total_discarded,
            "repaired": report.total_repaired,
            "coref_failed": len(coref_failures),
            "errors": len(errors),
        })

        save_ingest_report(
            sample_id=entry.sample_id,
            question=entry.question,
            extractor=ingestor.extractor_name,
            total_triples=report.total_triples,
            docs_processed=len(report.processed),
            docs_skipped=sum(1 for d in report.docs if d.skipped),
            zero_triple_docs=len(report.zero_triple_docs),
            errors=errors,
        )

        logger.info(
            "   %d triple tenute (%d riparate), %d scartate, %d scritte "
            "su Neo4j, %.1fs%s%s",
            report.total_extracted, report.total_repaired,
            report.total_discarded, report.total_triples,
            time.time() - t0,
            f" — COREF FALLITA su {coref_failures}" if coref_failures else "",
            f" — ERRORI: {errors}" if errors else "",
        )

    logger.info("═══ Totale (%.1fs) ═══", time.time() - t_start)
    for key in ("questions", "docs", "skipped", "extracted", "repaired",
                "discarded", "written", "coref_failed", "errors"):
        logger.info("  %-13s %d", key, totals[key])
    if discard_reasons:
        logger.info("  Scarti per motivo: %s",
                    dict(discard_reasons.most_common()))
    if client is not None:
        logger.info("  Grafo: %s", client.stats())
        client.close()


if __name__ == "__main__":
    main()
