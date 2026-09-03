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

## Phase 14: Fix perf — modelli ricaricati ad ogni click (2026-09-01)
- [x] `ClaimAttributor.__init__` (claim_attributor.py:70-71): `self._extractor` settato al filtro stringa poi SUBITO sovrascritto da `TripleExtractor()` — il filtro `extractor` (rebel/deepseek) non arrivava mai a `exact_match`/`semantic_fallback`/`query_partial`. Rinominato in `self._extractor_filter` (stringa) + `self._parser` (TripleExtractor).
- [x] `app.py`: `GraphWriter(client=neo4j)` e `ClaimAttributor(...)` istanziati a ogni click (write/verify), MAI dietro `@st.cache_resource` → SentenceTransformer + REBEL ricaricati da disco ad ogni interazione (causa dello stall "pytorch_model.bin" segnalato dall'utente). Aggiunti `get_graph_writer()` e `get_attributor(threshold, extractor)` cached; call site aggiornati.
- [x] fastcoref/transformers pin (`<4.56`) già presente in `requirements.txt` non committato — verificare `pip install -r requirements.txt` per allineare l'ambiente.
- [x] **Root cause vero stall "pytorch_model.bin"**: `app.py` settava `TRANSFORMERS_CACHE=D:\hf_home\transformers`, secondo cache root separato da `HF_HUB_CACHE` (`D:\hf_home\hub`, dove il modello era già completo). Ogni sessione risolveva nel dir sbagliato/vuoto, trovava un blob `.incomplete` residuo e riscaricava tutto da zero (1.6GB REBEL + f-coref). Rimossa la riga in app.py; aggiunto `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` in settings.py (override con `HF_ALLOW_ONLINE=1`).
- [x] **fastcoref era ANCORA no-op silenzioso**: `coref_resolver.py` chiamava `.get_resolved_text()`, metodo inesistente su `CorefResult` di fastcoref 2.1.6 (esistono solo `get_clusters()`/`get_logit()`) — ogni chiamata cadeva nell'except e ritornava testo non risolto. Riscritto `_resolve_clusters()`: ricostruisce il testo risolto da `get_clusters(as_strings=False)`, sostituendo ogni menzione con la più lunga del cluster. Verificato: "He" → "Tenma" su frase di test.

## Phase 15: Pipeline ibrida REBEL+DeepSeek — debug variants A/B (2026-09-01)

Contesto: la pipeline non e' piu' "REBEL *oppure* DeepSeek" ma ibrida. REBEL rende
poco -> serve misurare quanto contribuisce davvero (possibile rimozione futura).
Neo4j fuori scopo: solo JSONL + UI.

### Nuovi moduli
- [x] `src/ingestion/guardrails.py` — scarto triple corrotte, un motivo per tripla:
      `no_predicate`, `empty_subject`, `empty_object`, `subject_equals_object`,
      `no_span`, `span_not_verbatim`, `subject_not_in_span`, `object_not_in_span`,
      `subject_is_claim`, `object_is_claim` (+ `duplicate` a valle).
- [x] `span_matcher.anchor_span` / `anchor_span_aligned` — finestra di 1-2 frasi che
      contiene S e O a livello di content token. `anchor_span_aligned` ancora sul testo
      coref-RISOLTO e restituisce la finestra corrispondente sull'ORIGINALE: lo span
      salvato resta verbatim, ma "He" nell'originale non fa scartare la tripla.
- [x] `src/ingestion/hybrid_extractor.py` — varianti A (correct, 1 call/passaggio) e
      B (2 passate, 2 call/passaggio) + conteggi REBEL.
- [x] `src/llm/json_repair.py` — recupero JSON malformato dalle risposte LLM
      (array non chiuso, virgole finali) con parse per-oggetto come ultima risorsa.
      Collegato a `parse_response` (triple) e `parse_verdicts` (verdetti).
- [x] `output_store.save_hybrid_report()` — `triples_hybrid.jsonl`,
      `triples_hybrid_discarded.jsonl` (+ `discard_reason`), `hybrid_runs.jsonl`.
      Nessun campo `extractor`; resta `origin` = `deepseek` | `rebel_confirmed`.

### UI (Streamlit)
- [x] Tab "Hybrid Debug (REBEL + DeepSeek)": scelta domanda, varianti A/B (anche
      entrambe sullo stesso input coref), tabella per passaggio + totali
      `triple prodotte | sopravvissute | scartate | REBEL prodotte | REBEL confermate |
      REBEL rigettate | triple finali da REBEL | LLM calls | sec`, tabella dei motivi
      di scarto, confronto A vs B, dettaglio per passaggio (originale, coref, REBEL
      grezze, passata 1 DeepSeek, sopravvissute con span+origin, scartate, REBEL
      rigettate con motivo).

### Verifiche
- [x] Test stub (REBEL/DeepSeek finti): span verbatim, guardrail su tutte le classi
      di corruzione, invariante `rebel_matched + rebel_rejected == rebel_raw`.
- [x] Test live (REBEL reale + DeepSeek reale, passaggio ALCE 6669150, entrambe le
      varianti): A 16 triple / REBEL 9 -> 3 confermate, B 11 triple / REBEL 9 -> 2
      confermate. Tutti gli span verbatim sul testo originale.
- [x] `AppTest` headless: 3 tab, nessuna eccezione, Neo4j offline tollerato.

### Aperto
- [ ] Variante B: due triple con lo stesso (S, O) e predicati sinonimi (DeepSeek +
      REBEL) sopravvivono entrambe — la dedup e' su (S, P, O). Serve decidere se
      deduplicare per coppia o affidarsi al validatore.
- [ ] Batch su N domande per una statistica REBEL affidabile (qui: 1 passaggio).
- [ ] Scrittura Neo4j della pipeline ibrida (rimossa la proprieta' `extractor`).

### Fuori scopo
- Scrittura Neo4j (nessuna modifica a graph_writer/neo4j_client).

## Phase 16: Fix pipeline ibrida + esperimento A/D su 10 domande (2026-09-02)

Contesto: il primo giro aveva 38/153 span vuoti (chiesti al modello), nodi
generici tipo "game" con 14 archi, 10 triple con deittici irrisolti, contatori
REBEL incoerenti e due run non riproducibili sullo stesso input.

### FIX
- [x] **FIX 1 — span dalla pipeline.** Unita' di lavoro = la frase: coref sul
      passaggio, split, `align_sentences(originale, risolto)`. `claim_span` =
      frase ORIGINALE allineata; il campo span e' stato TOLTO dallo schema JSON
      chiesto al modello. Il motivo di scarto `no_span` non esiste piu'.
- [x] **FIX 2 — guardrail dove servono.** `entity_not_in_sentence` (match
      normalizzato + fuzzy leggero su morfologia/accenti), `generic_node`
      (NER/POS spaCy: niente named entity, niente PROPN, niente numero/data ->
      scarto), `unresolved_reference` (pronomi, "that same year", "the
      subsequent game"), `subject_equals_object`, `no_predicate`.
- [x] **FIX 3 — coref rumorosa.** `CoreferenceResolver` solleva
      `CorefUnavailable` invece di degradare; `check()` come health check; il
      runner si ferma con exit code 2 se il coref non carica. Pin
      `transformers>=4.41,<4.56` gia' presente e verificato (4.55.4 in venv).
- [x] **FIX 4 — contatori.** Ogni candidato REBEL ha uno e un solo stato
      (`confirmed` | `validated` | `rejected_llm` | `rejected_guardrail`)
      assegnato per indice. Un "keep" del validatore poi ucciso dai guardrail
      NON conta come confermato. Invariante testata.
- [x] **FIX 5 — riproducibilita'.** Cache LLM su disco in `DeepSeekClient`,
      chiave = SHA256 di (modello, temperatura, max_tokens, messaggi), log
      HIT/MISS, `cache_stats()` in UI e nel runner. `LLM_CACHE=0` per forzare
      chiamate fresche.

### Esperimento
- [x] `scripts/run_hybrid_experiment.py` — 10 domande ASQA, varianti A e D,
      REBEL eseguito una volta sola e condiviso, chiamate LLM parallele.
- [x] Variante A: frase + vocabolario dei predicati REBEL (NON le triple).
      Il vocabolario si ricava dall'output REBEL sul corpus: `rebel-large` e'
      un BART seq2seq, il suo `id2label` e' `LABEL_0/1/2` e non contiene le
      relazioni.
- [x] Variante D: DeepSeek cieco per frase -> confronto programmatico su
      (subject, object) -> una sola chiamata di validazione per i candidati
      REBEL residui. Variante B eliminata.
- [x] `src/ingestion/hybrid_analysis.py` — conferma REBEL per predicato, archi
      per nodo, diff A vs D.
- [x] `tests/test_hybrid_pipeline.py` — 22 test, nessuna rete.
- [x] UI: tab riscritto su A/D con le tre tabelle nuove + stato cache LLM.

### Fuori scopo
- Neo4j (nessuna scrittura, nessuna modifica a graph_writer/neo4j_client).

### Risultati del giro (10 domande ASQA, 50 passaggi, 267 frasi, 2026-09-02)

Vocabolario REBEL ricavato dall'output: 117 predicati distinti.

| | A (vocabolario) | D (cieca + validazione) |
|---|---|---|
| triple prodotte | 434 | 772 |
| sopravvissute ai guardrail | 308 | 565 |
| scartate | 126 | 207 |
| REBEL prodotte | 865 | 865 |
| REBEL confermate | 96 (**11.1%**) | 238 (**27.5%**) |
| triple finali da accordo | 81 | 89 |
| triple finali solo REBEL (validate) | 0 | 122 |
| LLM calls | 267 | 317 |

- Scarti A: generic_node 65, duplicate 37, entity_not_in_sentence 20,
  unresolved_reference 2, subject_equals_object 2. Zero `no_span`.
- Scarti D: generic_node 93, duplicate 43, entity_not_in_sentence 36,
  subject_equals_object 15, no_predicate 15, unresolved_reference 5.
- Diff A/D: 219 comuni, 82 solo A, 322 solo D. Il vocabolario REBEL RIDUCE la
  resa: A produce meno triple e con predicati piu' poveri (es. perde
  "Josef Bican | birth date | 25 September 1913").
- Conferma REBEL per predicato (D): `participating team` 66.7%,
  `located in the administrative territorial entity` 59.3%, `location` 58.8%,
  `inception` 54.2%, `performer` 50.0% vs `subclass of` 0%, `sport` 0%,
  `instance of` 4%, `date of birth` 15.4%. La media aggregata nasconde questa
  spaccatura: REBEL e' utile solo su un sottoinsieme di relazioni.

### Riproducibilita verificata
- Run 1 e run 2 divergevano di 2 triple in D: race sulla cache (due thread con
  lo stesso prompt, entrambi in miss, due risposte diverse dall API). Fix:
  single-flight in `DeepSeekClient` + test in `tests/test_llm_cache.py`.
- Run 2 vs run 3 (cache calda, 584 hit / 0 miss): summary, tabella predicati,
  archi per nodo, motivi di scarto e diff A/D **identici**. I numeri riportati
  qui sopra sono quelli di run 2/3.

### Aperto
- [ ] Decisione su REBEL: tenerlo solo per le relazioni con conferma > ~50%,
      oppure eliminarlo (costa ~20s/passaggio su CPU e porta 0 triple esclusive
      in A, 122 in D).
- [ ] I nodi-calamita residui sono entita' legittime (topic della domanda):
      "The Sound of Silence" 28 archi su 5 passaggi. Serve una regola sui nodi
      con nome ambiguo ("Louise", 19 archi su 5 passaggi) — probabile lavoro
      di entity linking, non di guardrail.

## Canonicalizzazione pre-write + separazione UI esperimenti (2026-09-03)

### Parte 1 — terza fase: extract -> canonicalize -> write
- [x] `src/ingestion/entity_canonicalizer.py`: cascata a 4 stadi
      (normalizzazione / lessicale / linking / embedding), scope parametrico
      `per_passage | per_question | global` (default `per_question`).
- [x] `surface_form` conservata sull'arco (`subject_surface`/`object_surface`),
      `external_id` sul nodo (`_MERGE_TRIPLE` aggiornata).
- [x] Embedding dei predicati spostato da GraphWriter alla canonicalizzazione
      (un batch per domanda; GraphWriter lo calcola solo se manca).
- [x] Log `data/outputs/canonicalization.jsonl` — una riga per OCCORRENZA di
      menzione (`output_store.save_canonicalization`).
- [x] `scripts/analyze_canonicalization.py` — stadi, merge, archi per nodo,
      nodi prima/dopo.
- [x] `tests/test_entity_canonicalizer.py` — 31 test, nessuna rete.
- [x] `merge_entity_into_canonical` + `EntityClusterer` marcati legacy.
- [x] UI: lo Step 6 canonicalizza prima di scrivere e mostra il riepilogo.

### Parte 2 — UI
- [x] `src/ui/resources.py` (risorse cached condivise) + `src/ui/experiments.py`
      con `render()`.
- [x] Tab esperimenti solo con `SHOW_EXPERIMENTS=1` (default nascosto).
- [x] Documentato in `.env.example` e `regole_progetto.md`.
- [x] `tests/test_ui_modes.py` — AppTest su entrambe le modalita'.

### Review
- Pipeline: `extract_entry` -> `canonicalize_entry` -> `write_entry`.
  `write_entry` non calcola piu' nulla: riceve triple canonicalizzate e gia'
  embeddate.
- Prova end-to-end su dati ALCE-like (5 triple, 3 passaggi, 1 domanda):
  `Josef "Pepi" Bican (25 Sept 1913)`, `Bican`, `Bican's record` e
  `Josef Bican` collassano su `Josef Bican` (external_id
  `wikipedia:Josef Bican`); `VanDeWeghe` -> `Kiki VanDeWeghe`;
  `Soccer Statistics Foundation (RSSSF)` -> `Soccer Statistics Foundation`.
  10 menzioni, 10 -> 7 nodi, **100% chiuse agli stadi 1-3** (7 / 2 / 1 / 0).
  La percentuale sul corpus vero va misurata con lo script, non assunta.
- `app.py`: 1114 -> 754 righe.
- Test: 56 passati. Restano 7 fallimenti PRE-ESISTENTI in
  `tests/test_hybrid_pipeline.py`, tutti `ModuleNotFoundError: spacy`
  nell'ambiente usato per i test (guardrail `generic_node`), non toccati da
  queste modifiche.

### Aperto
- [ ] Ri-tarare `ENTITY_CLUSTER_THRESHOLD` (0.90) per lo stadio 4 sui dati
      reali: dopo gli stadi 1-3 gli restano solo i casi difficili, quindi la
      soglia attuale e' un'ipotesi da verificare sul log.
- [ ] Il grafo esistente NON e' migrato: va rigenerato per avere
      `surface_form`/`external_id` sugli archi e sui nodi.

## REBEL fuori dalla pipeline principale (2026-09-03)

Estrattore unico: DeepSeek. REBEL resta solo negli esperimenti (il loro scopo
e' proprio misurare quanto aggiunge — vedi i numeri della fase ibrida sopra).

- [x] `settings`: `AVAILABLE_EXTRACTORS = ["deepseek"]`, `ACTIVE_EXTRACTOR`
      default `deepseek`. `EXTRACTOR_REBEL` resta come ETICHETTA per gli archi
      dei grafi vecchi, che restano interrogabili.
- [x] `build_extractor`: costruisce solo DeepSeek; con `"rebel"` alza un
      ValueError che indirizza agli esperimenti.
- [x] `ClaimAttributor`: il claim si parsa con `DeepSeekExtractor` (simmetria di
      estrazione, `regole_progetto.md` §4) + errore esplicito se la chiave API
      manca, invece di "nessuna tripla estratta".
- [x] UI: via il radio dell'estrattore nello Step 1 e il selettore di grafo nel
      tab Claim Attribution; `get_debug_rebel_extractor` spostato da
      `src/ui/resources.py` a `src/ui/experiments.py` (unico consumatore).
- [x] `.env` / `.env.example`: `ACTIVE_EXTRACTOR=deepseek` — con `rebel` la
      pipeline avrebbe scritto archi `deepseek` e l'attribution avrebbe
      interrogato solo quelli `rebel` (grafo pieno, zero risultati).
- [x] `triple_extractor.py` marcato "fuori dalla pipeline": resta perche' ci
      vive anche la NamedTuple `Triple`, usata da tutto il sistema.
- [x] `tests/test_pipeline_extractor.py` — 7 test contro il rientro silenzioso
      di REBEL.

### Aperto
- [ ] Il grafo scritto con `extractor="rebel"` non e' migrato: va rigenerato con
      DeepSeek (o interrogato di proposito con `ACTIVE_EXTRACTOR=rebel`).
