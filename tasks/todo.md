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

## Phase 10: Refactor legacy → cartella legacy/ (2026-07-18) — COMPLETATA
- [x] Spostati in `legacy/`: orchestrator.py, vector_retriever.py, wiki_retriever.py, highlight_renderer.py (git mv, storia preservata)
- [x] Creato `legacy/legacy_settings.py`: CERCA_*, SUPPORT_THRESHOLD_*, CHROMA_*, EMBEDDING_MODEL (bge-m3), prompt anime (CHAIN_OF_CITATION/RESUME/REFINEMENT) — rimossi da config/settings.py
- [x] Import legacy aggiornati: `from legacy import legacy_settings as settings` (ri-esporta anche settings attivi)
- [x] src/generator/__init__.py: rimosso InGenerationOrchestrator; llama_generator.py: default stop_tag hardcoded (LlamaGenerator resta attivo — usato da question_parser)
- [x] Rimossi package vuoti src/retriever/ e src/visualization/
- [x] Verifica REBEL English-only: rebel-large, SRC_LANG=None, test EN → triple corrette; test IT → triple degradate/errate (conferma: corpus deve essere inglese)
- [ ] TODO futuro: rimuovere ingestione PDF/TXT, lavorare direttamente su domande ALCE (integrazione successiva)

## Phase 9: Audit critico (LLM Council, 2026-07-11) — FASE 1 COMPLETATA
- [x] Council 5 advisor + peer review + sintesi → `AUDIT.md`, `council-report-2026-07-11.html`, `council-transcript-2026-07-11.md`
- [ ] Decisione utente su fix autorizzati (F1–F14 in AUDIT.md, ordinati per severità)
- Fix critici individuati: F1 tabella proprietà predicati (pilastro B assente — coseno-solo matcha relazioni inverse), F2 buco pred_emb=None→verified 1.0, F3 harness valutazione inesistente, F4 scoping percorso domande

## Phase 11: Corpus ALCE + estrattori multipli (2026-08-03)
- [x] `src/ingestion/alce_loader.py` — entry ASQA + primi 5 docs, nessun chunking, solo `text` originale
- [x] `src/ingestion/alce_ingestor.py` — orchestratore skip → coref → estrazione → claim_span → MERGE → registro
- [x] `src/ingestion/deepseek_extractor.py` — estrattore LLM (API OpenAI-compatible, JSON mode) + prompt
- [x] `src/ingestion/processed_registry.py` — `data/processed_ids.txt`, gestisce i doc a ZERO triple
- [x] `src/ingestion/span_matcher.py` — `claim_span` ancorato al testo originale
- [x] Neo4j: `MERGE` con chiave `(predicate, source_id, extractor)`, indice, `is_source_ingested`, `ingested_source_ids`, `triples_by_source`, `delete_by_source`, `stats_by_extractor`
- [x] Filtro `extractor` in `exact_match` / `semantic_fallback` / `query_partial` + `ClaimAttributor`
- [x] `merge_entity_into_canonical` preserva `source_id`/`extractor`/`claim_span`
- [x] UI: tab "Corpus ALCE" (lista domande ✓/◐/○, ingest, testo originale → coref → triple REBEL vs DeepSeek), selettore grafo nel tab attribution
- [x] Rimossi ingestione PDF/TXT (`document_loader.py`, PyMuPDF, CHUNK_SIZE_WORDS) — chiude il TODO di Phase 10
- [x] `.env.example` + caricamento `.env` in `config/settings.py`
- [x] Verificato headless: loader (948 entry), span matcher, parser DeepSeek, registro, skip/force/zero-triple (stub), `AppTest` senza eccezioni

### Da verificare con Neo4j acceso (non testabile con DB offline)
- [ ] MERGE idempotente: doppia ingestione della stessa domanda → conteggio archi invariato
- [ ] Due estrattori sullo stesso passaggio → archi separati, `stats_by_extractor` li distingue
- [ ] Attribution filtrata per estrattore restituisce solo archi di quel grafo
- [ ] Entity clustering non mescola i grafi (props preservate)

### Prossimo
- [ ] Ingestione batch (CLI) su N domande per il confronto REBEL vs DeepSeek
- [ ] Metrica di copertura per estrattore usando `answers_found` come ground truth

## Phase 12: Ollama → legacy (2026-08-03)
- [x] `llama_generator.py` e `OLLAMA_*` spostati in `legacy/` (git mv, storia preservata)
- [x] `src/generator/`, `src/retriever/`, `src/visualization/` rimossi (package vuoti)
- [x] `src/llm/deepseek_client.py` — trasporto HTTP condiviso (chat + JSON mode + retry)
- [x] `QuestionParser` portato da Ollama a DeepSeek (altrimenti il percorso "domanda" moriva in silenzio)
- [x] `DEEPSEEK_TEMPERATURE = 0.0` su tutte le chiamate
- [x] UI: rimosso selettore modello Ollama e badge; resta il badge DeepSeek
- [x] Verificato live: ping API, estrazione su passaggio ALCE, question parser IT+EN, AppTest senza eccezioni

## Phase 13: Timing extract() REBEL (2026-08-23)
- [x] `DocResult.extract_seconds` in `alce_ingestor.py` — timer isolato su `extractor.extract()`, esclusi coref/embedding/write
- [x] UI: caption "⏱ extract() REBEL: Xs · N triple · source_id · chunk_index" solo per estrattore REBEL, dopo ingestione
- Device REBEL già auto-cuda se disponibile (`TripleExtractor._load()`), nessuna modifica necessaria
- [x] Bug: `ingest_btn` falliva con `ServiceUnavailable` se Neo4j offline (is_processed → client lazy-fails). Aggiunto bottone "🧪 Debug extract REBEL (no DB, no write)" in `app.py` — usa `get_debug_rebel_extractor()`/`get_debug_coref_resolver()` (cache_resource dedicati, no Neo4jClient), ancora claim_span con `span_matcher.best_span` in locale, mostra triple/tempo/passaggio senza toccare il grafo
