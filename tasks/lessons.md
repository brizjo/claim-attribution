# Lessons Learned

## Architecture

- 0.0s coref bug root cause: fastcoref was simply NOT installed. Silent try/except hid the ImportError. Fix: install fastcoref + log errors instead of swallowing.
- coreferee max Python 3.11 — dead on Python 3.12. spacy-experimental needs MSVC build tools on Windows. fastcoref 2.1.6 is pure Python and works on 3.12.
- Entity clustering in Neo4j (EntityClusterer) is cross-document safety net — fastcoref handles within-doc pronoun resolution only, not cross-doc aliases.
- fastcoref 2.1.6 breaks with transformers >= 4.56 (`'FCorefModel' object has no attribute 'all_tied_weights_keys'`): FCorefModel calls `init_weights()` instead of `post_init()`, and the 4.56 tied-weights refactor sets `all_tied_weights_keys` only in `post_init()`. Pin `transformers>=4.41.0,<4.56` (4.55.4 works). A `<5.0.0` pin is NOT enough. Failure is a silent coref skip (logged warning only).
- regole_progetto.md explicitly overrides in-generation pipeline — new system is LPG/Neo4j only
- REBEL model input max ~256 tokens → chunk docs at 200 words with 50-word overlap
- REBEL is BART seq2seq. transformers 5.x removed `text2text-generation` pipeline alias → use `AutoModelForSeq2SeqLM` + `AutoTokenizer` + manual `model.generate()`. Use `max_new_tokens` not `max_length`.
- REBEL decode: keep `skip_special_tokens=False` so `<triplet>`/`<subj>`/`<obj>` markers survive for the parser.
- Default model = `Babelscape/rebel-large` (English BART, ~1.5GB, fast). To switch multilingual: set `REBEL_MODEL = "Babelscape/mrebel-large"` + `REBEL_SRC_LANG = "it_IT"`. `TripleExtractor` auto-detects mREBEL by name and applies mBART tokenizer config (src_lang/tgt_lang/decoder_start_token_id=tp_XX). Single regex parser handles both REBEL `<subj>/<obj>` markers and mREBEL `<ENTITY_TYPE>` typed markers — any bracketed token acts as segment separator.
- mBART tokenizer requires `sentencepiece` package — not pulled by transformers automatically.
- REBEL/mREBEL extracts triples from declarative statements only. Questions ("Chi è X?") must NOT be sent to mREBEL — route through Ollama-based QuestionParser instead, which yields a partial triple (?, P, O) / (S, ?, O) / (S, P, ?) for graph pattern query.
- Same embedding model for ingestion AND retrieval is a hard rule (parallel to classical RAG): predicate_embedding stored on edge during ingest is compared against query predicate embedding produced by the same `PREDICATE_EMBEDDING_MODEL`.
- Ingestion is per-chunk streaming (coref → mREBEL → embed → Neo4j write per chunk), NOT batch-at-end. Document node finalized once per file via `GraphWriter.finalize_document()`. Crash mid-file preserves already-written triples + marks document `status="error"`. Trade-off: lose mREBEL batching speedup; gain durability + live progress.
- Coreference resolution is MANDATORY (no skip option) — required for triple quality.
- `chunk_text` stored on Neo4j RELATES_TO edge MUST be the ORIGINAL chunk text, not the coref-resolved version. mREBEL receives resolved text for better extraction; users must see source evidence verbatim. After `extractor.extract([resolved])`, run `triples = [t._replace(chunk_text=original_text) for t in triples]`.
- mREBEL ingest = MINIBATCH (size = `REBEL_BATCH_SIZE`), not per-chunk. Per-chunk wastes the batched forward pass; Document node still finalized once at end. Crash mid-file → already-written batches persist.
- mREBEL generation MUST set `num_return_sequences=3` (matching `num_beams=3`) — default keeps only top hypothesis, losing 2/3 of candidate triples. Dedupe per chunk on (subject, predicate, object) lowercased.
- mREBEL output emits special tokens WITHOUT surrounding whitespace (e.g. `tp_XX<triplet>` glued together). Parser must use regex `re.sub(r"(<[^>]+>)", r" \1 ", text)` to space-separate before `split()` — otherwise `<triplet>` is never matched as its own token and triple count collapses to ~0.
- Store all RELATES_TO with predicate as property (not as relationship type) — Cypher type names can't have spaces
- Same REBEL model for both ingestion and claim parsing (consistency requirement from regole_progetto.md §4)
- Predicate embeddings stored as float array on relationship — cosine similarity computed in Python (relationship vector indexes require Neo4j 5.18+)
- Neo4j multi-DB: server may host more than one database. `Neo4jClient` MUST be constructed with `database=...` (default `"neo4j"`, settable via `NEO4J_DATABASE` env). All sessions go through `_session()` so the DB is consistent. UI shows active DB name + list of visible DBs — if Browser shows different counts, pick the matching DB in Browser.
- `batch_write_triples` uses explicit `tx.commit()`/`tx.rollback()` (not `with begin_transaction()` auto-commit) — driver behavior was version-dependent, explicit is bulletproof.
- Streamlit perf: any object wrapping a HF model (`SentenceTransformer`, `TripleExtractor`, `CoreferenceResolver`) MUST be built inside a `@st.cache_resource` function, never directly in a button-click branch — a plain `ClassName(...)` call in the script body re-runs on every rerun/click and reloads the weights from disk each time. Symptom looked like a re-download (progress bar on `pytorch_model.bin`) but was actually a repeated `from_pretrained()` load. Caught in `ClaimAttributor`/`GraphWriter` (app.py) — both were constructed fresh per click; fixed with `get_attributor()`/`get_graph_writer()` cached getters.
- Copy-paste bug pattern: `self._x = a; self._x = b` (reassigning the same attribute name to two different things) silently discards `a`. Found in `ClaimAttributor.__init__` — the `extractor` provenance-filter string was clobbered by the `TripleExtractor()` instance one line later, so the rebel/deepseek graph filter was never applied to any attribution query. Always grep for the attribute name after adding a second assignment to catch this.
- **fastcoref was STILL a silent no-op after the transformers<4.56 pin** (2026-09-01): `coref_resolver.py` called `preds[0].get_resolved_text()`, a method that does not exist on fastcoref 2.1.6's `CorefResult` (only `get_clusters()`/`get_logit()` exist) — every call hit the broad except, logged a warning, returned unresolved text. Fixed by building resolved text from `get_clusters(as_strings=False)` (char spans): per cluster, replace every mention with that cluster's longest span (proper nouns beat pronouns), applied right-to-left to keep offsets valid. **Lesson: a broad `except Exception: log + fallback` around a third-party call hides API-signature drift, not just transient errors — verify the library's actual public API (read the source, not memory) before trusting a silent-fallback path is "working."**
- **Split HF cache root caused repeated full re-downloads** (2026-09-01): `app.py` set `os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_home\transformers"` at the very top of the file (before `config/settings.py` is even imported) — a *second*, separate cache directory from `HF_HUB_CACHE` (`D:\hf_home\hub`), which is exactly the trap `config/settings.py`'s own comment warns about (and had already fixed on the settings.py side, but not in app.py — the fix didn't propagate to every place env vars are set). Every session, `from_pretrained` resolved into the empty/stale second dir, found an old `.incomplete` blob there, and re-triggered a full download (1.6GB REBEL, biu-nlp/f-coref) — looked like "re-downloading a model that's already cached," and stalled for minutes when the freshness-check network call hung. Fix: removed the stray line from app.py; added `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` in settings.py (opt out via `HF_ALLOW_ONLINE=1` env var) so a fully-cached model never touches the network again. **Lesson: when the same env-var block exists in two files, a fix applied to one silently doesn't apply to the other — grep for every place `os.environ[...]` sets the same HF/cache var before trusting a cache-path fix is complete.**
- Never set the deprecated `TRANSFORMERS_CACHE` env var alongside `HF_HUB_CACHE`/`HF_HOME` — they point to two separate cache roots (`hf_home/transformers` vs `hf_home/hub`), and `AutoTokenizer.from_pretrained` reads the legacy one. A file left mid-write there (e.g. process killed during a stuck download) reads back as a corrupt/blank JSON forever with no fallback to the good copy — this was the actual cause of the fastcoref "Expecting value: line 1 column 1" load failure (`biu-nlp/f-coref` tokenizer_config.json was 393 bytes of blank spaces in the `transformers` cache dir while the `hub` copy was intact). Fix: set only `HF_HOME` + `HF_HUB_CACHE`/`HUGGINGFACE_HUB_CACHE`, never `TRANSFORMERS_CACHE` ([config/settings.py](../config/settings.py)).

## Fase 11 — Corpus ALCE + estrattori multipli (2026-08-03)

### Modifiche al database
- **Chiave logica dell'arco `RELATES_TO`: `(predicate, source_id, extractor)`.**
  Tutte le scritture usano `MERGE`, mai `CREATE` (`Neo4jClient._MERGE_TRIPLE`):
  la re-ingestione dello stesso passaggio non duplica archi, ma la stessa
  relazione vista in due passaggi diversi resta come due archi distinti
  (due prove indipendenti).
- Proprietà nuove sull'arco: `source_id` (= `doc["id"]` ALCE, provenienza),
  `extractor` (`"rebel"` | `"deepseek"`), `claim_span` (frase del chunk
  ORIGINALE che supporta la tripla).
- Indice `rel_source_extractor` su `(r.source_id, r.extractor)` per rendere
  O(1) il check di idempotenza (creazione in try/except: richiede Neo4j 5.x).
- `merge_entity_into_canonical` copiava le proprietà con `CREATE` elencandole
  a mano → perdeva `source_id`/`extractor`/`claim_span` e avrebbe **mescolato
  i grafi dei due estrattori** al primo clustering. Ora usa `MERGE` con la
  stessa chiave logica. *Lezione: ogni proprietà nuova sull'arco va aggiunta
  anche nei due blocchi di copia del clustering, altrimenti sparisce in
  silenzio.*

### Modifiche alla logica di matching
- `exact_match`, `semantic_fallback`, `query_partial` accettano
  `extractor: Optional[str]` → `WHERE ($ext IS NULL OR r.extractor = $ext)`.
  `ClaimAttributor(extractor=...)` lo propaga a tutte e tre.
  **Solo filtro di provenienza: scoring e routing invariati.** `None` = nessun
  filtro. I due grafi non vanno mai interrogati insieme: predicati identici da
  estrattori diversi falserebbero il confronto.

### Idempotenza a due livelli (nessuno dei due basta da solo)
- Neo4j (`is_source_ingested`) **non vede i documenti a zero triple**: senza
  archi risulterebbero non processati e verrebbero riestratti a ogni run —
  ed è l'estrazione la parte costosa, non il MERGE.
- `ProcessedRegistry` (`data/processed_ids.txt`, TSV
  `extractor \t source_id \t n_triples \t timestamp`) copre quel caso e
  fornisce la metrica di copertura (`zero_triple_ids()`), ma non sa se il
  grafo è stato svuotato.
- Un doc è saltato se risulta processato in ALMENO uno dei due; `force=True`
  cancella gli archi (`delete_by_source`) e riestrae.
- `Clear Graph` NON tocca il registro: dopo uno svuotamento serve `force`.

### claim_span
- Né REBEL né DeepSeek danno offset affidabili sul testo originale (REBEL non
  ne dà, DeepSeek vede il testo coref-risolto). `span_matcher.best_span()`
  ancora la tripla alla frase del testo ORIGINALE con massimo overlap
  lessicale su subject/object (+ span dell'LLM come cue), split a regex.
  Deterministico, zero costo, nessun modello caricato.
- `chunk_text` resta sempre il testo originale ALCE: l'evidenza dev'essere
  verbatim, mai il testo riscritto dalla coref.

### Corpus
- `summary` ed `extraction` nei doc ALCE sono generati da un LLM a monte:
  **ignorati**, si usa sempre `text`.
- I passaggi sono già chunk (~100 parole, top-5 ri-rankati oracle) → nessun
  chunking. `document_loader.py` (PDF/TXT) rimosso, insieme a
  `CHUNK_SIZE_WORDS`/`CHUNK_OVERLAP_WORDS` e alla dipendenza PyMuPDF.

### Streamlit
- `st.stop()` dentro un tab ferma l'INTERO script, non il tab: il corpo del
  tab ALCE è una funzione (`_render_ingest_tab`) così i `return` di guardia
  (corpus mancante, filtro vuoto) non spengono il tab di attribution.

### Segreti
- `DEEPSEEK_API_KEY` letta da `.env` nella root (già in `.gitignore`) da
  `config/settings.py`; `python-dotenv` è opzionale — c'è un parser di
  fallback per non rompere gli ambienti che non l'hanno. Template in
  `.env.example`. Mai hardcodare la chiave nei sorgenti.

## Fase 12 — Ollama fuori dalla pipeline attiva (2026-08-03)

- `src/generator/llama_generator.py` → `legacy/llama_generator.py`; package
  `src/generator/` rimosso (era vuoto). `OLLAMA_*` spostati da
  `config/settings.py` a `legacy/legacy_settings.py`. `legacy/orchestrator.py`
  importa da `legacy.llama_generator`.
- **`QuestionParser` dipendeva da Ollama**: spostare il generator senza altro
  avrebbe ucciso in silenzio il percorso "domanda" dell'attribution (il parser
  ritorna `QuerySpec()` vuoto su eccezione → sembra solo "domanda non
  parsabile"). Portato su DeepSeek. *Lezione: prima di archiviare un modulo,
  cercare chi lo importa — qui il fallimento sarebbe stato silenzioso.*
- Trasporto HTTP DeepSeek estratto in `src/llm/deepseek_client.py`
  (`chat(messages, json_mode)`): lo usano sia l'estrattore di triple sia il
  question parser, invece di duplicare retry/auth. `DeepSeekExtractor` tiene
  solo prompt + parsing.
- `temperature=0.0` per TUTTE le chiamate DeepSeek (estrazione e parsing):
  output deterministico, mai creativo.
- JSON mode DeepSeek richiede che la parola "JSON" compaia nel prompt —
  vincolo dell'API, entrambi i prompt la contengono.
- Ollama non era una dipendenza pip (chiamate via `requests`): da
  `requirements.txt` non c'era nulla da togliere, solo commenti da correggere.
- Verificato live con la chiave reale: ping API ok, 10 triple estratte da un
  passaggio ALCE con `claim_span` verbatim, question parser IT+EN corretto.
