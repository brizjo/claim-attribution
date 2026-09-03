from .alce_loader import AlceEntry, AlceLoader
from .alce_ingestor import AlceIngestor, DocResult, IngestReport, build_extractor
from .coref_resolver import CoreferenceResolver, CorefUnavailable
from .deepseek_extractor import DeepSeekExtractor
from .entity_canonicalizer import (
    CanonicalizationResult,
    EntityCanonicalizer,
    Mention,
    Resolution,
    build_linker,
    normalize_mention,
)
from .graph_writer import GraphWriter
from .hybrid_extractor import (
    HybridExtractor,
    HybridTriple,
    PassageResult,
    RebelCandidate,
    RunReport,
    SentenceUnit,
    VARIANTS,
    VARIANT_A,
    VARIANT_D,
    VARIANT_LABELS,
    rebel_vocabulary,
)
from .output_store import (
    save_coref,
    save_triple,
    save_triples_batch,
    save_attribution,
    save_ingest_report,
    save_hybrid_report,
    save_canonicalization,
    load_jsonl,
)
from .processed_registry import ProcessedRegistry
# `TripleExtractor` (REBEL) non e' piu' nella pipeline: resta esportato per gli
# esperimenti.  `Triple` invece e' il tipo di tripla di tutto il sistema.
from .triple_extractor import Triple, TripleExtractor

__all__ = [
    "AlceEntry",
    "AlceLoader",
    "AlceIngestor",
    "DocResult",
    "IngestReport",
    "build_extractor",
    "CoreferenceResolver",
    "CorefUnavailable",
    "DeepSeekExtractor",
    "CanonicalizationResult",
    "EntityCanonicalizer",
    "Mention",
    "Resolution",
    "build_linker",
    "normalize_mention",
    "GraphWriter",
    "HybridExtractor",
    "HybridTriple",
    "PassageResult",
    "RebelCandidate",
    "RunReport",
    "SentenceUnit",
    "VARIANTS",
    "VARIANT_A",
    "VARIANT_D",
    "VARIANT_LABELS",
    "rebel_vocabulary",
    "ProcessedRegistry",
    "Triple",
    "TripleExtractor",
    "save_coref",
    "save_triple",
    "save_triples_batch",
    "save_attribution",
    "save_ingest_report",
    "save_hybrid_report",
    "save_canonicalization",
    "load_jsonl",
]
