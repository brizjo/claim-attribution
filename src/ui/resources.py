"""
Risorse Streamlit condivise fra i tab dell'app.

Stanno qui e non in `app.py` perche' `@st.cache_resource` e' identificata dalla
funzione: se due moduli definissero due getter uguali, i modelli verrebbero
caricati DUE volte.  Un solo posto, un solo modello in memoria.

Qui stanno solo le risorse della pipeline principale (DeepSeek, coref, Neo4j,
canonicalizer, attribution).  REBEL non e' piu' una risorsa condivisa: vive in
`src/ui/experiments.py`, l'unico modulo che lo usa.
"""

from __future__ import annotations

import streamlit as st


@st.cache_resource
def get_neo4j_client():
    try:
        from src.graph.neo4j_client import Neo4jClient
        return Neo4jClient()
    except Exception:
        return None


@st.cache_resource
def get_alce_loader():
    """ALCE corpus loaded once per session (~10MB)."""
    from src.ingestion.alce_loader import AlceLoader
    return AlceLoader()


@st.cache_resource
def get_ingestor(extractor_name: str, use_coref: bool = True):
    """Un ingestor per (estrattore, coref) — tiene caldi i client fra le run.

    Il resolver e' CONDIVISO (`get_debug_coref_resolver`): fastcoref viene
    caricato una volta sola anche se esistono piu' ingestor cached."""
    from src.ingestion.alce_ingestor import AlceIngestor, build_extractor
    return AlceIngestor(
        client=get_neo4j_client(),
        extractor=build_extractor(extractor_name),
        resolver=get_debug_coref_resolver(),
        use_coref=use_coref,
    )


@st.cache_resource
def get_deepseek():
    from src.ingestion.deepseek_extractor import DeepSeekExtractor
    return DeepSeekExtractor()


@st.cache_resource
def get_debug_coref_resolver():
    from src.ingestion.coref_resolver import CoreferenceResolver
    return CoreferenceResolver()


@st.cache_resource
def get_generator():
    """Generatore grounded (studio <claim, context>, stadio 1) — client caldo."""
    from src.generation.answer_generator import AnswerGenerator
    return AnswerGenerator()


@st.cache_resource
def get_graph_writer():
    """One GraphWriter — keeps its SentenceTransformer loaded across writes."""
    from src.ingestion.graph_writer import GraphWriter
    return GraphWriter(client=get_neo4j_client())


@st.cache_resource
def get_attributor(semantic_threshold: float, extractor: str):
    """One ClaimAttributor per (threshold, extractor) — keeps the encoder loaded."""
    from src.attribution.claim_attributor import ClaimAttributor
    return ClaimAttributor(
        client=get_neo4j_client(),
        semantic_threshold=semantic_threshold,
        extractor=extractor,
    )


@st.cache_resource
def get_canonicalizer():
    """
    Canonicalizzatore delle entita' (fase fra estrazione e scrittura).
    Cached: tiene caldo l'encoder e, con scope `global`, lo stato accumulato.
    """
    from src.ingestion.entity_canonicalizer import EntityCanonicalizer
    return EntityCanonicalizer()
