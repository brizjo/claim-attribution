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

## Fase 15 — Pipeline ibrida REBEL + DeepSeek (2026-09-01)

- **Lo span va ancorato sul testo coref-RISOLTO, salvato dal testo ORIGINALE.**
  Il modello vede "Barack Obama served as..." mentre l'originale dice "He served
  as...": pretendere subject/object dentro lo span *originale* scarterebbe le
  triple corrette. `anchor_span_aligned` cerca la finestra sul risolto e riporta
  la stessa finestra sull'originale per indice di frase (la coref sostituisce
  menzioni in-place, il numero di frasi si conserva); se i due split hanno un
  numero di frasi diverso si ricade sull'ancoraggio diretto sull'originale.
- **Il JSON mode di DeepSeek NON garantisce JSON valido.** In un run reale il
  validatore ha restituito `...}}` invece di `...}]}`: con `json.loads` secco
  tutti e 9 i verdetti sparivano e TUTTE le triple REBEL risultavano "rigettate"
  — cioe' esattamente la metrica che serve a decidere se tenere REBEL, falsata
  al 100% senza un solo errore a schermo. `src/llm/json_repair.py` chiude le
  parentesi rimaste aperte e, come ultima risorsa, parsa i singoli oggetti
  bilanciati. *Lezione: un parser LLM che ritorna [] su errore va sempre letto
  come "quanti record ho perso in silenzio?", non come "il modello non ha
  prodotto nulla".*
- **La provenienza non si chiede all'LLM, si calcola.** Il flag `from_rebel` del
  prompt e' inaffidabile: `origin` viene dal confronto (subject, object) a
  livello di content token tra candidati REBEL e set finale. Il predicato e'
  escluso dalla chiave perche' l'LLM lo *corregge* e la tripla resta la stessa
  proposta di REBEL.
- **Contare le triple REBEL "tenute" sulle triple FINALI sbaglia i totali.** Due
  candidati REBEL con lo stesso (S, O) e predicati diversi collassano in una
  sola tripla finale: `rebel_kept` (triple finali con origin rebel) e' <=
  `rebel_matched` (candidati REBEL confluiti). L'invariante da testare e'
  `rebel_matched + rebel_rejected == rebel_raw`, e nell'accounting il candidato
  gia' bocciato dal validatore va controllato PRIMA di quelli tenuti, altrimenti
  finisce in entrambi i conteggi.
- `extractor` non e' piu' una proprieta' della tripla: la pipeline e' una sola.
  Nei JSONL ibridi resta solo `origin` (`deepseek` | `rebel_confirmed`), utile
  per contare offline il contributo di REBEL.
- Guardrail: predicato senza content token ("is a") = nessuna asserzione, si
  scarta. Stessa regola per subject/object composti solo da stopword ("it"):
  senza content token la tripla non e' ancorabile a nessuno span.

## Fase 16 — Fix pipeline ibrida + esperimento A/D (2026-09-02)

- **Lo span non si chiede al modello: lo sa gia' la pipeline.** Chiedere
  `claim_span` nel JSON produceva 38 span vuoti su 153 e altrettante triple
  valide scartate. Lavorando frase per frase la sorgente e' nota: si assegna
  `claim_span` = frase ORIGINALE allineata a quella coref-risolta data
  all'estrattore (`span_matcher.align_sentences`). Dopo il fix: 870 triple
  salvate, 0 span vuoti, 0 span non verbatim, motivo `no_span` sparito.
- **I nodi generici sono il vero generatore di falsi positivi.** Non serve una
  regola sul testo ma su cosa il nodo *identifica*: NER/POS della frase, e se
  l'entita' non tocca ne' una named entity ne' un PROPN ne' un numero/data ->
  `generic_node`. E' il guardrail che scarta di piu' (65 triple in A, 93 in D):
  il nodo "game" che collezionava 14 archi da partite diverse non nasce piu'.
- **Un "keep" del validatore non e' una conferma.** Se poi i guardrail uccidono
  la tripla, il candidato REBEL va contato fra le rigettate. Prima veniva
  marcato `validated` e saltato in fase di riconciliazione: i totali non
  quadravano (`matched + rejected != prodotte`). Regola: ogni candidato riceve
  uno e un solo stato, assegnato per indice alla fine, e nessuno stato
  intermedio salta la riconciliazione finale.
- **temperature=0 non e' riproducibilita'.** Due run identici davano 14 e 16
  triple. La cache su disco in `DeepSeekClient` (SHA256 di modello +
  temperatura + max_tokens + messaggi, log HIT/MISS, `cache_stats()`) rende il
  ri-run identico e gratuito. Attenzione: in variante A il vocabolario REBEL fa
  parte del prompt, quindi cambiare l'insieme di domande cambia i prompt e
  invalida la cache — e' corretto, ma va saputo.
- **`rebel-large` non ha un `id2label` utile.** E' un BART seq2seq: il config
  espone `LABEL_0/1/2`. Il "vocabolario REBEL" si ricava dall'output del
  modello sul corpus (117 predicati distinti su 50 passaggi): e' il vocabolario
  che REBEL usa davvero, non quello che dichiara.
- **fastcoref NON era rotto in questo ambiente** (transformers 4.55.4, pin
  `<4.56` gia' presente): `check()` risolve "He" -> "Barack Obama". I 10
  riferimenti deittici nei dati non venivano da un coref saltato ma da un suo
  limite: risolve le catene di menzioni, non "that same year" / "the subsequent
  game". Quelli li prende il guardrail `unresolved_reference`. Il resolver ora
  solleva comunque `CorefUnavailable` invece di degradare in silenzio, e il
  runner esce con codice 2.
- **Il tasso di conferma aggregato e' una media che mente.** Su D: 27.5%
  complessivo, ma `participating team` 67%, `location` 59%, `inception` 54%
  contro `subclass of` 0%, `sport` 0%, `instance of` 4%. La decisione su REBEL
  va presa per relazione, non sul totale.
- **Dare a DeepSeek il vocabolario di REBEL peggiora la resa** (variante A: 308
  triple finali contro 562 di D, 320 triple presenti solo in D). Il vocabolario
  chiuso ancora l'estrazione a relazioni Wikidata-style e fa perdere fatti che
  la frase dice esplicitamente.
- **Una cache concorrente senza single-flight non basta a rendere un run
  riproducibile.** Ri-eseguito lo stesso esperimento a cache calda: 584 hit / 0
  miss (quindi prompt identici e risposte identiche), eppure la variante D
  dava 772 triple invece di 770. Causa: nel primo run 6 thread lavoravano in
  parallelo e alcune frasi identiche comparivano in passaggi diversi — due
  thread mancavano la cache sullo STESSO prompt, chiamavano entrambi l'API e
  ricevevano risposte diverse (l'API non e' deterministica nemmeno a
  temperature=0). La cache conservava poi solo l'ultima scritta, quindi il
  secondo run vedeva una risposta sola. Fix: single-flight in `DeepSeekClient`
  (`_claim_key`/`_release_key`): il primo thread chiama, gli altri con la
  stessa chiave aspettano e leggono dalla cache. Test dedicato in
  `tests/test_llm_cache.py`. *Lezione: in un esperimento parallelo la cache va
  progettata come "una chiamata per chiave", non come "scrivi quando hai
  finito" — altrimenti la nondeterminicita' dell'API rientra dalla finestra.*

## Canonicalizzazione delle entita' (2026-09-03)

- **Il posto giusto per unificare i nodi e' PRIMA della scrittura.** Durante
  l'estrazione il modello vede una frase alla volta e non puo' sapere che
  "VanDeWeghe" e' "Kiki VanDeWeghe"; dopo la scrittura, unire nodi gia' in
  Neo4j (`merge_entity_into_canonical`) significa riattaccare archi a mano e
  perdere proprieta'. La pipeline ha ora tre fasi: `extract_entry` ->
  `canonicalize_entry` -> `write_entry`, con `write_entry` ridotto a I/O.
- **Lo scope della canonicalizzazione e' una scelta con un costo, quindi e' un
  parametro.** `per_passage` e' troppo stretto (`Josef Bican` in 3 passaggi
  della stessa domanda resterebbe 3 nodi: recall persa in attribution);
  `global` e' troppo largo (`Louise`, 19 archi su 5 passaggi, collassa persone
  diverse) e in piu' regala recall usando evidenza fuori dallo scope della
  domanda — **se si usa `global` va dichiarato come limite in tesi**. Default
  `per_question`: la domanda ALCE con i suoi 5 passaggi.
- **Il genitivo sassone si riduce al possessore, non si scarta.**
  `Wilt Chamberlain's set` e `Campbell's call` non sono entita', ma
  `Wilt Chamberlain` e `Campbell` lo sono: scartare la menzione butterebbe via
  l'arco insieme al rumore. La riduzione scatta solo se dopo `'s` c'e' un altro
  token — cosi' `Levi's` e `McDonald's`, dove l'apostrofo fa parte del nome,
  restano intatti.
- **Lo strip dell'articolo iniziale ha un costo noto e accettato.**
  `The Sound of Silence` -> `Sound of Silence`: la normalizzazione e' applicata
  a TUTTE le menzioni, quindi le forme con e senza articolo si unificano invece
  di restare due nodi. Il prezzo e' che il titolo perde il suo articolo: la
  forma verbatim resta comunque su `surface_form`.
- **`title` del passaggio ALCE e' entity linking gratuito.** E' il titolo
  dell'articolo Wikipedia da cui il passaggio e' preso: aggancia l'entita'
  principale senza rete, senza soglia e senza ambiguita' quando un solo titolo
  dello scope contiene la menzione (se ne combaciano due — "Louise" fra
  "Louise Brown" e "Louise Smith" — non si aggancia niente e si lascia decidere
  allo stadio 4). Da' anche il NOME canonico: meglio il titolo Wikipedia della
  menzione piu' lunga vista nel testo.
- **Alternative di linking valutate.** *DBpedia Spotlight*: implementato come
  linker opzionale (`ENTITY_LINKER=spotlight`) perche' e' un servizio HTTP
  senza installazione, ma e' rete a ogni menzione e il servizio pubblico e'
  spesso lento — resta OPT-IN e degrada esplicitamente (log + stadio 4), mai in
  silenzio (precedente: fastcoref). *spaCy `entityLinker`*: richiede il
  download di un KB (~1.5GB) e un modello aggiuntivo, sproporzionato per un
  segnale che il campo `title` gia' copre. *Wikipedia search API*: una query
  per menzione, e sui nomi comuni restituisce il risultato piu' popolare, cioe'
  esattamente il falso merge che si vuole evitare. Scelta: `title` come
  default.
- **Il tie-break alfabetico sceglie il refuso.** Con `max(..., key=(len, form))`
  fra `Cristiano Ronaldo` e `Cristiano Ronalod` (stessa lunghezza) vince
  l'errore, perche' 'o' > 'd'. La forma canonica ora si sceglie per
  `(lunghezza, frequenza, prima occorrenza)`: mai per ordine alfabetico.
- **Il log e' per OCCORRENZA, non per forma.** Una riga per menzione con il suo
  `source_id`: senza questo non si puo' contare quanti archi ha un nodo e su
  quanti passaggi distinti, che e' il segnale per trovare i falsi merge
  (`scripts/analyze_canonicalization.py`). Se la maggioranza delle menzioni si
  chiude agli stadi 1-3, la canonicalizzazione e' in gran parte deterministica:
  e' la frase da scrivere in tesi, ma va misurata, non assunta.
- **Gli esperimenti non sono funzionalita'.** Il tab ibrido A/D e' uscito da
  `app.py` (1114 -> 754 righe) ed e' in `src/ui/experiments.py`, visibile solo
  con `SHOW_EXPERIMENTS=1`. Le risorse Streamlit condivise stanno in
  `src/ui/resources.py`: due `@st.cache_resource` con lo stesso corpo ma in
  moduli diversi sono due cache diverse, cioe' REBEL caricato due volte.

## Fase 17 — Guardrail in pipeline + repair + Neo4j batch (2026-09-03)

- **I guardrail erano solo negli esperimenti: la pipeline principale scriveva
  su Neo4j le stesse classi di corruzione misurate là** (subject=object, nodi
  generici, deittici, entita' inventate). Il run live lo conferma: 11 triple
  su 51 bocciate alla prima domanda ingerita. *Lezione: un fix validato in un
  esperimento non esiste finche' non e' portato nel percorso di produzione —
  e i due percorsi vanno tenuti UNO.*
- **La UI ri-implementava l'estrazione inline** (`app.py` Step 4: passaggio
  intero all'estrattore, niente sentence split, niente guardrail, best_span
  senza allineamento): il percorso UI e il percorso batch producevano triple
  DIVERSE dallo stesso input. Ora entrambi passano da
  `AlceIngestor.extract_doc`. *Lezione: se un bottone della UI e uno script
  batch fanno "la stessa cosa" con due implementazioni, una delle due e' gia'
  sbagliata — cercare sempre la duplicazione del flusso, non solo del codice.*
- **Le triple bocciate non si buttano: si riparano.** Il repair round (1 call
  DeepSeek per frase fallita, col motivo dello scarto tradotto in istruzione
  correttiva) recupera i fatti con la forma rotta; le riparate ripassano gli
  STESSI guardrail e chi fallisce due volte muore (`stage="repair"` nel log).
  Nel run live DeepSeek ha correttamente lasciato cadere le irreparabili
  (0 riparate su 11): il costo e' basso perche' si paga solo per le frasi con
  almeno uno scarto, e la cache LLM copre i ri-run.
- **Un bug di fastcoref su UN passaggio non deve uccidere il passaggio.**
  `extract_doc` ora degrada al testo originale marcando `coref_failed`, e il
  guardrail `unresolved_reference` scarta i deittici rimasti: prima l'intero
  passaggio finiva in `error` e si perdevano anche le triple buone. Il
  fail-fast globale (fastcoref non carica affatto) resta nel runner batch
  (exit 2): degradare per-passaggio e' sicuro SOLO perche' ora i guardrail
  fanno da rete a valle.
- **`getattr(extractor, "client", None)` tiene il Protocol pulito**: il repair
  serve solo all'estrattore LLM; stub dei test ed estrattori sperimentali
  senza `.client` saltano il repair senza cambiare l'interfaccia `Extractor`.
- **Neo4j (2026-09-03)**: istanza Desktop "RAG", `bolt://localhost:7687`,
  database `neo4j`, credenziali in `.env`. 40 triple scritte -> 34 archi: la
  differenza sono MERGE collassati DOPO la canonicalizzazione (stessa chiave
  `(predicate, source_id, extractor)` fra gli stessi nodi canonici) — e'
  idempotenza che lavora, non triple perse. Ri-run: 5/5 skip, grafo invariato.

## REBEL fuori dalla pipeline principale (2026-09-03)

- **Decisione presa sui numeri, non a sensazione.** L'esperimento ibrido esisteva
  per rispondere a "REBEL serve?". Risposta sui 50 passaggi / 267 frasi: REBEL
  conferma l'11.1% delle triple in variante A e il 27.5% in D; dare a DeepSeek
  il vocabolario chiuso di REBEL **peggiora** la resa (308 triple finali contro
  562, con fatti espliciti persi come `Josef Bican | birth date | 25 Sept 1913`);
  costo ~20s per passaggio su CPU. Estrattore unico: DeepSeek.
- **Togliere un estrattore non e' cancellare un modulo.** `Triple` (la
  NamedTuple usata da TUTTO il sistema) vive in `triple_extractor.py` insieme a
  `TripleExtractor` (REBEL): il modulo resta. Quel che cambia e' chi lo
  costruisce — `build_extractor` non lo istanzia piu' e alza un ValueError che
  indirizza agli esperimenti, invece di sparire e lasciare un `KeyError` oscuro.
- **La simmetria di estrazione va spostata insieme all'estrattore.**
  `ClaimAttributor` parsava il claim con `TripleExtractor()`: lasciarlo li'
  avrebbe verificato claim estratti da REBEL contro un grafo scritto da
  DeepSeek — vocabolari diversi, cioe' il requisito §4 rotto in silenzio. Ora
  parsa con `DeepSeekExtractor`, con un test che lo blocca
  (`tests/test_pipeline_extractor.py`).
- **`DeepSeekExtractor.extract` logga e salta i chunk falliti.** Nel percorso di
  attribution quel comportamento traduceva "API key mancante" in "nessuna
  tripla estratta dal claim": un fallimento silenzioso travestito da risultato.
  Aggiunto un check `is_available()` a monte con messaggio esplicito.
- **`ACTIVE_EXTRACTOR` e' un filtro sul grafo, non solo una scelta di modello.**
  Restava `rebel` nel `.env` locale: la pipeline avrebbe scritto archi
  `deepseek` e l'attribution avrebbe interrogato solo quelli `rebel` — grafo
  pieno e zero risultati. Cambiato in `deepseek` in `.env` e `.env.example`.
  L'etichetta `EXTRACTOR_REBEL` resta in settings: i grafi vecchi restano
  interrogabili puntando `ACTIVE_EXTRACTOR=rebel` di proposito.
