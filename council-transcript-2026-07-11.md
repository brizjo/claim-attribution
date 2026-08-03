# LLM Council Transcript — 2026-07-11

## Original question

Full Phase-1 audit request for the master's thesis codebase (graph-based claim attribution, Neo4j LPG): does the code actually support the three pillars — (A) edge provenance, (B) correct relational matching (equivalent predicates match, inverse relations must not), (C) ingestion/verification extraction symmetry? Produce a critical diagnosis with severity-ranked flaws and proposed (not implemented) fixes; stop before touching code.

## Framed question

Master's thesis codebase (D:\python_projects\rag\rag-claim-attribution): graph-based claim attribution on Neo4j LPG. Ingestion: corpus → chunks → coref → REBEL triple extraction → Neo4j edges carrying chunk_text/source_file/predicate embedding. Verification: claim → same extractor → exact match, then cosine fallback on predicate embedding (S/O fixed). Three pillars: (A) edge provenance, (B) correct relational matching, (C) extraction symmetry. Does the codebase actually support the pillars? Which flaws could invalidate the thesis, ranked critical/medium/cosmetic? Stakes: empirical core of a master's thesis.

## Anonymization mapping (peer review)

| Letter | Advisor |
|---|---|
| A | The Executor |
| B | The First Principles Thinker |
| C | The Contrarian |
| D | The Outsider |
| E | The Expansionist |

---

## Advisor responses

### The Contrarian (Response C)

No. Two of the three pillars fail structurally, and one failure is exactly the false-positive path a committee member would find in ten minutes.

**CRITICAL — Pillar B is not implemented, it's contradicted.** The fallback is *pure cosine at 0.75* on the predicate string (claim_attributor.py:151-166; neo4j_client.py:342-355; settings.py:51). There is no inverse-relation guard, no NLI check, nothing. MiniLM embeds antonymous relation labels ("parent of"/"child of", "date of birth"/"date of death") at ~0.85+ cosine. Stored triple `(A, parent_of, B)`, claim `(A, child_of, B)`: same subject/object, same direction, cosine above threshold → `verified=True`. The thesis says "cosine alone insufficient"; the code is cosine alone. This single path invalidates the contribution as stated.

**CRITICAL — unit-confidence hole in the question path.** claim_attributor.py:216-219: if the parser returns no predicate, the *first arbitrary graph row* is returned as `verified=True, similarity=1.0`. Worse, query_partial pools rows via `LIMIT $pool` with no ORDER BY (neo4j_client.py:386-389) — nondeterministic evidence presented at fabricated 100% confidence.

**CRITICAL — Pillar C symmetry is false for questions.** Questions never touch REBEL: they go through an Ollama JSON parser emitting free-form, often Italian predicates ("protagonista", question_parser.py:47-63), matched against REBEL's English Wikidata-style labels. Different extractor, different vocabulary — the symmetry claim holds only for the declarative half.

**MEDIUM — Pillar A: provenance destruction on entity merge.** merge_entity_into_canonical copies edges with `WHERE target <> canon` (neo4j_client.py:238, 256), then `DETACH DELETE dup` (269-272): every dup↔canon edge — and its chunk_text evidence — is silently destroyed. Clustering at 0.90 (entity_clusterer.py:89-91) can also merge distinct entities, attributing chunks to the wrong subject.

**MEDIUM:** only `triples[0]` of an arbitrary beam is verified (claim_attributor.py:131); semantic_fallback's docstring claims "both directions" but the Cypher is directed (neo4j_client.py:317 vs 325-327); CLAUDE.md/settings.py:5 claim Llama-3 coref, code uses fastcoref — thesis text won't match code.

**COSMETIC:** hardcoded password (settings.py:33); duplicate edges from 50-word chunk overlap inflating graph stats.

### The First Principles Thinker (Response B)

Strip the framing: this system doesn't verify claims. It verifies that **two entity strings co-occur on some edge whose predicate embedding is nearby**. That's a co-mention test wearing a verification costume. From that first principle, the flaws rank themselves.

**CRITICAL — Pillar B is not implemented, only asserted.** The thesis says "cosine alone insufficient" for inverse relations; the code is *exactly* cosine alone (claim_attributor.py:151-157, neo4j_client.py:342-355). Direction is enforced in the Cypher — but direction only protects you when the inverse fact is stored in inverse orientation. Claim "(Anna, parent of, Marco)" vs stored edge "(Anna, child of, Marco)": same direction, MiniLM cosine("parent of","child of") ≈ 0.85 > threshold 0.75 → `verified=True`. No antonym/inverse mechanism exists anywhere in the repo. This is the structural false-positive path a committee will find. Bonus red flag: the docstring claims "both directions" (neo4j_client.py:316) while the query is one-directional (:325-327) — the author doesn't know what their own matcher does.

**CRITICAL — question path rubber-stamps.** With no predicate, *any* edge between S and O returns `verified=True, similarity=1.0` (claim_attributor.py:216-219). Pure co-mention.

**CRITICAL — Pillar C symmetry is false.** Ingestion runs coref rewriting and entity clustering (app.py:305, 371-375); verification feeds the raw claim as a dummy chunk (claim_attributor.py:110) and matches literal `normalized_name` — no cluster lookup. Aliases silently become false negatives; same extractor ≠ same pipeline.

**MEDIUM — Pillar A breaks post-clustering.** `merge_entity_into_canonical` drops edges *between* duplicate and canonical (`WHERE target <> canon`, neo4j_client.py:238, 256; then `DETACH DELETE`, :270) — provenance destroyed for precisely the facts linking merged entities. Also: `triples[0]` arbitrarily picked (claim_attributor.py:131); `CREATE` not `MERGE` duplicates edges on re-ingest (:90).

**COSMETIC** — settings/UI say mREBEL, config ships English-only rebel-large (settings.py:42).

The right question isn't "does the code support the pillars" — it's whether pillar B was ever designed at all. It wasn't.

### The Expansionist (Response E)

The council is underselling what's already sitting in this repo. Three undervalued assets:

**1. Direction is your inverse-relation defense — and nobody's claiming it.** The docstring at `src/graph/neo4j_client.py:317` says "both directions," but the Cypher at :325-327 matches only `(sub)-[r]->(obj)`, directed. That "bug" is your pillar-B thesis argument in disguise: inverse relations expressed with swapped arguments ("Y child_of X") structurally *cannot* match "X parent_of Y" — the LPG topology filters them before cosine ever runs. Fix the docstring, keep the behavior, run an ablation (directed vs. undirected fallback) and you've converted cosine's known weakness into a measurable structural contribution. That's a chapter, not a footnote.

**2. Free corroboration counting.** `batch_write_triples` uses `CREATE`, not `MERGE`, on relationships (`neo4j_client.py:90`), so the same fact asserted in N chunks yields N parallel edges, each with its own `chunk_text`/`source_file`. Right now `exact_match` throws this away with `LIMIT 1` (:299). Return them all and you get multi-source evidence and redundancy-weighted confidence — zero new infrastructure, dramatically stronger attribution story.

**3. Two dormant baselines already in-repo.** `src/attribution/matrix.py` is a deterministic FActScore-style lexical auditor, and `config/settings.py:67-69` reveals a full Chroma vector-RAG stack. That's a ready-made three-way head-to-head (graph vs. vector vs. lexical) — the comparative evaluation examiners love, at near-zero cost.

Plus: pillar C is *architecturally enforced*, not incidental — `ClaimAttributor` instantiates the same `TripleExtractor` (`claim_attributor.py:64`) and same embedder setting as `GraphWriter` (`graph_writer.py:21`). Swap the verification extractor in an ablation, show the accuracy collapse, and symmetry becomes an empirical finding rather than an assumption. The question path (`query_partial`) is quietly a second contribution: partial-triple KGQA with provenance. This thesis is bigger than its author thinks.

### The Outsider (Response D)

**Verdict: Pillar B is not implemented at all, and Pillars A and C hold only conditionally.**

**CRITICAL**

1. **Pillar B is exactly the thing the thesis says is insufficient.** The semantic fallback is pure cosine against a 0.75 threshold — nothing else (`claim_attributor.py:150-166`, `settings.py:51`). There is zero inverse-relation handling anywhere; grep for "inverse/antonym/direction" hits only one line, and that line is a **false docstring**: `neo4j_client.py:316` says "both directions" while the Cypher at :325-327 is strictly directed. MiniLM cosine("parent of", "child of") will clear 0.75, so a false inverse claim on the same S→O pair gets `verified=True`. An examiner testing this one case sinks the pillar.

2. **Pillar C is violated on the question path.** Questions are parsed by an Ollama LLM, not the REBEL extractor (`claim_attributor.py:174-196`) — a different extractor for verification. Worse, when no predicate is extracted, *any* edge between the entities is returned as `verified=True, similarity=1.0` (`claim_attributor.py:216-219`). Manufactured confidence.

3. **The method description doesn't match the code.** Spec/settings say coreference = Ollama/Llama-3 (`settings.py:24-28`); the UI says "spaCy+coreferee" (`app.py:205, 408`); the actual code uses fastcoref (`coref_resolver.py:25`). Also, default extractor is English-only `rebel-large` (`settings.py:42`) while the whole UI and question grammar are Italian. If the thesis text mirrors any of these, it misdescribes its own system.

**MEDIUM**

- Pillar A holds only via `app.py:319-325`, which restores verbatim chunk text; `GraphWriter` alone stores coref-rewritten text. Provenance lives in UI code.
- Entity merging silently deletes dup↔canonical edges (`neo4j_client.py:238, WHERE target <> canon` + `DETACH DELETE`) — lost provenance.
- Only `triples[0]` of a multi-triple claim is checked (`claim_attributor.py:131`); re-ingestion duplicates edges (`CREATE`, `neo4j_client.py:90`).

**COSMETIC**: hardcoded DB password (`settings.py:33`); unjustified 0.75/0.90 thresholds; dead legacy prompt code.

### The Executor (Response A)

Verdict: A holds (mostly), C holds for claims, B is not implemented at all. Monday-morning plan below.

**B — nothing exists.** Grep for "inverse|symmetric" across the repo: zero hits. The only relational logic is cosine ≥ 0.75 (`settings.py:51`) in `semantic_fallback` (`neo4j_client.py:308`). MiniLM puts "father of"/"son of" around 0.8 — that's a false-positive machine, and the docstring at `neo4j_client.py:316-317` even lies ("both directions") while the Cypher at :325-327 is directed-only. So symmetric facts ("spouse") stored one way are false negatives too.

**A — one real hole.** Edges carry `chunk_text`/`source_file` fine, but `merge_entity_into_canonical` (`neo4j_client.py:238`, `:256`) has `WHERE target <> canon` / `WHERE source <> canon`, then `DETACH DELETE dup` (:270): every edge between the duplicate and the canonical entity is silently destroyed — provenance gone. Also `batch_write_triples` uses `CREATE` (:90), so re-ingest duplicates edges.

**C — claim path is genuinely symmetric** (same `TripleExtractor` at `claim_attributor.py:64`, same MiniLM via `settings.PREDICATE_EMBEDDING_MODEL` on both sides). The question path (`question_parser.py`, Ollama free-text predicates like "protagonista") breaks it — scope questions out of the thesis claims, don't fix.

**Ordered fixes, smallest first:**
1. `settings.py`: add `PREDICATE_PROPERTIES = {canonical: {"aliases": [...], "inverse": ..., "symmetric": bool}}` (~30 lines, from your corpus's actual REBEL predicates).
2. `claim_attributor._attribute_claim` (:150-167): before accepting a semantic candidate, reject if `(claim_pred, cand_pred)` is an inverse pair; accept immediately if same alias class. ~15 lines.
3. Same function: if predicate is symmetric or has an inverse, retry `exact_match`/`semantic_fallback` with S/O swapped (inverse-mapped predicate). ~10 lines.
4. `neo4j_client.py:238/256`: drop the `<> canon` filters, dedupe instead.
5. `neo4j_client.py:90`: `CREATE` → `MERGE`.

Days 1–3 = steps 1–3; that's the whole defense of pillar B. Test with 10 hand-written inverse/symmetric claim pairs before touching anything else.

---

## Peer reviews

### Reviewer 1

**1. Strongest: C.** It correctly ranks the pillar-B cosine-only hole as thesis-invalidating, verifies with a concrete failure case, and uniquely catches the nondeterminism compounding the unit-confidence bug (`LIMIT $pool` with no ORDER BY). I confirmed its line cites: `claim_attributor.py:216-219` does return `verified=True, similarity=1.0` when `pred_emb is None`, and the "both directions" docstring at `neo4j_client.py:316` contradicts the directed Cypher. B is close but overreaches — its "verification ignores clustering" claim needs evidence the others don't corroborate.

**2. Biggest blind spot: E.** Its central reframe is wrong: directedness only blocks inverses stored with *swapped* arguments. The dangerous case — claim "(A, child_of, B)" vs stored "(A, parent_of, B)", same direction — sails through cosine. E converts a critical false-positive path into a "chapter," which would actively mislead the author.

**3. All five missed: evaluation.** Nobody asked whether a labeled test set, metrics, or negative-claim benchmarks exist. A thesis dies without empirical validation regardless of code correctness — pillar B especially needs an inverse-claim test suite to be *measured*, not just implemented. Also unexamined: whether REBEL reliably extracts triples from short claim sentences at all (extraction recall bounds the whole system).

### Reviewer 2

**1. Strongest: Response C.** It has the best severity calibration and the widest verified coverage: the pure-cosine inverse hole (confirmed at claim_attributor.py:150-166), the unit-confidence rubber stamp (:216-219, confirmed), the un-ordered `query_partial` pool, the Italian/English vocabulary split on the question path, the merge-deletion provenance loss, and doc/code drift. B is close and adds the cluster-lookup asymmetry, but C ranks more precisely. A is the most actionable but wrongly waves the question path out of scope.

**2. Biggest blind spot: Response E.** Its centerpiece — "directed Cypher is the inverse defense" — only blocks inverses stored in *swapped* orientation. The same-direction antonym case sails through cosine, exactly the false positive B/C/D confirmed. E converts the thesis-sinking bug into a "chapter" without noticing it's still open, and calls CREATE-duplication "free corroboration" while ignoring re-ingest inflation.

**3. All five missed:** there is no evaluation harness for any pillar — the only tests (`tests/test_rag_rewardbench.py`, `test_bertscore.py`) target the legacy vector-RAG stack. Without a gold claim set and metrics, even fixed code can't defend the thesis empirically.

### Reviewer 3

I spot-checked the code before ranking: `semantic_fallback` is indeed directed-only with a false "both directions" docstring (neo4j_client.py:316-327), the question path fabricates `similarity=1.0` (claim_attributor.py:216-219), and only `triples[0]` is checked (:131). The council's factual claims held up.

**1. Strongest: C.** Every citation verified, correct severity ranking, and it uniquely caught the nondeterministic `LIMIT $pool` with no ORDER BY (neo4j_client.py:386). B deserves honorable mention for the one finding nobody else has: verification matches literal `normalized_name` and bypasses coref/entity clustering, so Pillar C's "same extractor" ≠ same pipeline.

**2. Biggest blind spot: E.** Its thesis — direction is the inverse-relation defense — fails on the realistic case: corpus stores (Anna, "mother of", Marco); false claim (Anna, "child of", Marco) has the *same* S→O direction, cosine clears 0.75, `verified=True`. E rebrands bugs as chapters without testing one adversarial pair.

**3. All five missed:** (a) there is no evaluation harness at all — no labeled claim set, no metrics, no threshold calibration; (b) `not_found` conflates "claim is false" with "graph lacks the fact" — fatal ambiguity for a hallucination-attribution thesis.

### Reviewer 4

**1. Strongest: B.** It nails the correct first principle (the system is a co-mention test), and it's the only response that catches the claim-path symmetry break: ingestion applies coref + entity clustering, but verification matches literal entity names with no alias/cluster lookup — I verified `claim_attributor.py` contains zero references to `normalized/alias/cluster`. A and E assert "C holds" and C limits the break to questions; B alone saw it applies to declarative claims too.

**2. Biggest blind spot: E.** Its centerpiece — directed Cypher as the inverse-relation defense — is wrong: an inverse predicate stated in the *same* S→O orientation passes direction and clears cosine. E rebrands the thesis-sinking flaw as a chapter. (E's corroboration-counting and dormant-baseline points are genuinely valuable, though; D's `app.py:319-325` verbatim-text restore is real — I confirmed it.)

**3. All five missed:** there is no evaluation harness. No gold set of equivalent/inverse/paraphrase claim pairs, no precision/recall computation anywhere. Even with every fix applied, none of the three pillars can be empirically defended — the committee will ask for numbers first.

### Reviewer 5

I spot-checked the contested code paths (claim_attributor.py:216-219, neo4j_client.py:308-398) — the critical claims in B/C/D are accurate.

**1. Strongest: C.** Full pillar coverage, verified-accurate line references, correct severity ordering, and two catches nobody else made: `query_partial` pools with `LIMIT` and no `ORDER BY`, and the Italian-Ollama vs English-REBEL vocabulary mismatch on questions. B is a close second — its coref/cluster-lookup asymmetry is a unique, real Pillar-C hole.

**2. Biggest blind spot: E.** Directed Cypher only blocks inverses stored in *swapped* orientation; the killer case — claim "(Anna, child of, Marco)" vs stored "(Anna, parent of, Marco)" — is same-direction, and cosine waves it through.

**3. All five missed:** there is no evaluation harness anywhere — every pillar is currently unfalsifiable. Relatedly: since claims and graph share REBEL's closed Wikidata vocabulary, exact match should dominate — nobody asked how often the risky cosine fallback actually fires. Measure that first.

---

## Chairman synthesis

See `AUDIT.md` (full Phase-1 formatted diagnosis) and `council-report-2026-07-11.html` (visual verdict).

**Where the council agrees (high confidence, cross-verified by reviewers):**
1. Pillar B is not implemented: the semantic fallback is pure cosine ≥ 0.75 with zero inverse/symmetric/transitive handling. Same-direction antonym predicates (stored "parent of", claimed "child of") produce `verified=True`. Thesis-sinking false-positive path.
2. `pred_emb is None` → `verified=True, similarity=1.0` on an arbitrary, nondeterministically selected edge (no ORDER BY). Fabricated confidence.
3. The "both directions" docstring is false; the Cypher is directed.
4. Entity merge destroys dup↔canonical edges and their provenance.
5. Method description drift: coref (Llama-3 vs coreferee vs fastcoref), mREBEL vs rebel-large.

**Where the council clashes:**
- Direction-as-defense (E) vs false-positive machine (B/C/D): reviewers unanimously sided against E for the same-direction case; but E's directed-vs-undirected ablation remains a good *experiment* once the inverse guard exists.
- Pillar C scope: Executor says claim path is symmetric (same extractor/embedder — true at model level); First Principles says the *pipeline* is asymmetric (coref+clustering at ingestion, raw literal match at verification — true at normalization level, verified). Both correct at different layers.
- Question path: fix it (C/D) vs scope it out of the thesis claims (A). Chairman: scope out or demote to "exploratory feature" — fixing costs more than it buys before the deadline, but the 1.0-confidence hole must be closed regardless.

**Blind spots caught only in peer review:**
- No evaluation harness exists for the graph method (all 5 reviewers). Pillars are unfalsifiable; committee will ask for numbers first.
- `not_found` conflates "claim is false" with "graph lacks the fact."
- Nobody measured how often the risky cosine fallback actually fires vs exact match.
- Extraction recall of REBEL on short claim sentences bounds the whole system and is untracked (silent false negatives).

**Recommendation:** implement the explicit predicate-property table (aliases/inverse/symmetric) with an inverse guard and symmetric/inverse retry; close the unit-confidence hole; then build the gold evaluation set (equivalent/inverse/paraphrase/negative pairs) and report precision/recall including fallback-fire rate. Fix provenance (merge edge-loss, original-text-on-edge) next. Scope the question path out of the core thesis claims.

**The one thing to do first:** write the ~30-pair gold set (10 true paraphrase, 10 inverse/antonym, 10 absent facts) and run it against the current system to measure the false-positive rate — it proves the bug today and proves the fix tomorrow, and the thesis needs those numbers either way.
