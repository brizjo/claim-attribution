# Claim Attribution — LPG/Neo4j Implementation

## Phase 1: Infrastructure
- [x] Create tasks/todo.md and tasks/lessons.md
- [x] Update config/settings.py (Neo4j, REBEL, embedding params)
- [x] Update requirements.txt
- [x] Create src/graph/neo4j_client.py

## Phase 2: Ingestion Pipeline
- [x] src/ingestion/document_loader.py (PDF/TXT → chunks)
- [x] src/ingestion/coref_resolver.py (Llama-3 coreference)
- [x] src/ingestion/triple_extractor.py (REBEL → triples)
- [x] src/ingestion/graph_writer.py (triples → Neo4j)

## Phase 3: Attribution
- [x] src/attribution/claim_attributor.py (exact match + semantic fallback)

## Phase 4: UI
- [x] Rewrite app.py (Streamlit — Ingest tab + Claim Attribution tab)

## Phase 5: spaCy Coref Refactor
- [x] Replace fastcoref → coreferee + en_core_web_lg in coref_resolver.py
- [x] requirements.txt: fastcoref → coreferee>=1.4.0
- [x] settings.py: SPACY_MODEL → en_core_web_lg
- [x] app.py labels updated

## Phase 6: Question Answering (Option B)
- [x] src/attribution/question_parser.py — Ollama LLM parses question → partial triple
- [x] Neo4jClient.query_partial() — Cypher with optional S/O filters + cosine on predicate emb
- [x] ClaimAttributor routes question vs claim
- [x] UI Tab 2 accepts both, highlights answer field on question mode

## Phase 7 (parked): Hybrid Dense + Graph Retrieval
Architectural note from user (2026-04-27):
- Triple DB alone CAN sustain claim attribution + question answering, but at scale a hybrid retrieval layer is needed.
- Plan: ChromaDB (already in legacy requirements) for dense vector recall on chunk_text, then Neo4j for graph verification.
- Same embedding model for ingestion + retrieval (parallel to classical RAG consistency principle, already enforced for predicate embeddings).
- Skip until Option B shows real-world limitations on the corpus.

## Phase 8: Verification
- [ ] Test full pipeline end-to-end with sample TXT file
- [ ] Verify Neo4j nodes/edges created correctly
- [ ] Verify claim attribution returns correct source chunk
- [ ] Check semantic fallback triggers when exact match fails
