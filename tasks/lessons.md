# Lessons Learned

## Architecture

- 0.0s coref bug root cause: fastcoref was simply NOT installed. Silent try/except hid the ImportError. Fix: install fastcoref + log errors instead of swallowing.
- coreferee max Python 3.11 — dead on Python 3.12. spacy-experimental needs MSVC build tools on Windows. fastcoref 2.1.6 is pure Python and works on 3.12.
- Entity clustering in Neo4j (EntityClusterer) is cross-document safety net — fastcoref handles within-doc pronoun resolution only, not cross-doc aliases.
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
