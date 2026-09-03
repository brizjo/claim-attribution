## 1. OBIETTIVO E SCOPO DEL PROGETTO
L'obiettivo è la realizzazione di un sistema di "Claim Attribution" oggettivo e deterministico. Il sistema deve superare i limiti dei RAG vettoriali puri utilizzando un Labeled Property Graph (LPG) come fonte di verità. 


## 2. ARCHITETTURA DEL SISTEMA E DATA PIPELINE

### Fase 1: Ingestione e Pre-processing del Corpus
Non sono caricati file dall'utente, il corpus è preso da un dataset già esistente (ASQA). Esso deve essere preprocessato, attraverso la divisione in chunk, la risoluzione delle coreferenze e infine l'estrazione delle triple, previa una divisione del passaggio in frasi semplici attraverso un sentence tokenizer.

**Estrattore unico: DeepSeek (2026-09-03).** REBEL è stato tolto dalla pipeline
principale. Motivo, misurato sui 50 passaggi dell'esperimento ibrido: REBEL
confermava l'11% (variante A) / 27.5% (variante D) delle triple, con un tasso
per relazione spaccato in due (`participating team` 67% contro `instance of` 4%,
`sport` 0%); e ancorare DeepSeek al vocabolario chiuso di REBEL **riduceva** la
resa (308 triple contro 562, con fatti espliciti persi). Il costo era ~20s per
passaggio su CPU. REBEL resta nel repo solo negli **esperimenti**, che servono
proprio a documentare questa decisione.

3.  **Mapping e Metadati:** Ogni relazione creata nel database deve includere come proprietà:
    * Il testo del chunk originale in chiaro.
    * Il riferimento al documento (nome file, autore, data).
    * L'embedding vettoriale della stringa del predicato.

**Guardrail + repair in pipeline (2026-09-03).** Ogni tripla estratta passa i
guardrail (`src/ingestion/guardrails.py`: subject=object, nodi generici,
deittici irrisolti, entita' non presenti nella frase, ...) DENTRO
`AlceIngestor.extract_doc`. Le bocciate NON si buttano subito: tornano a
DeepSeek — una chiamata per frase fallita, col motivo dello scarto
(`src/ingestion/triple_repair.py`) — e le riparate ripassano gli stessi
guardrail; chi fallisce due volte finisce in
`data/outputs/triples_discarded.jsonl`. Un fallimento di fastcoref sul singolo
passaggio degrada al testo originale (`coref_failed`) invece di perdere il
passaggio: i guardrail fanno da rete. UI e batch usano lo STESSO
`extract_doc`: mai ri-implementare l'estrazione inline.

**Ingestione batch:** `scripts/ingest_alce.py` (template della pipeline:
health check fail-fast -> extract -> guardrail+repair -> canonicalize ->
write). Neo4j: istanza Desktop "RAG", `bolt://localhost:7687`, database
`neo4j`, credenziali in `.env`.

### Fase 1-bis: Canonicalizzazione delle entita' (fra estrazione e scrittura)
La pipeline ha tre fasi, non due: `extract_entry` -> `canonicalize_entry` ->
`write_entry` (`src/ingestion/entity_canonicalizer.py`).  Il momento giusto per
unificare i nodi e' quando tutte le triple della domanda sono in memoria e
nessuna e' ancora scritta: durante l'estrazione il modello vede una frase alla
volta, dopo la scrittura unire nodi gia' in Neo4j e' fragile e perde proprieta'
(`merge_entity_into_canonical` resta ma e' legacy).

* **Scope** (`CANONICALIZATION_SCOPE`): `per_passage` | `per_question` |
  `global`, default `per_question` — e' un parametro perche' e' ablabile.
* **Cascata a 4 stadi**, il primo che risolve vince: (1) normalizzazione
  deterministica, (2) similarita' lessicale deterministica, (3) entity linking
  su ID esterno (il campo `title` del passaggio ALCE e' un titolo Wikipedia:
  linking gratuito e offline), (4) embedding + soglia coseno come ripiego.
* **Provenienza verbatim**: la forma canonica va sul nodo, la menzione
  originale resta sull'arco (`subject_surface` / `object_surface`).  L'ID
  esterno va sul nodo (`external_id`).
* **Embedding dei predicati**: calcolato qui, un batch per domanda; la
  scrittura si limita a scrivere.
* **Log**: `data/outputs/canonicalization.jsonl`, una riga per menzione
  (menzione, forma canonica, stadio, confidenza, source_id, sample_id,
  external_id).  Analisi: `scripts/analyze_canonicalization.py`.

### Fase 2: Database e Storage
* **Tecnologia:** Neo4j (Labeled Property Graph).
* **Struttura Nodi:** Entità normalizzate (Entity Linking/Clustering).
* **Struttura Archi:** Relazioni arricchite con metadati e vettori.

### Fase 3: Logica Ibrida di Claim Attribution (Fallback Semantico)
La verifica di un claim segue una gerarchia di precisione:
1.  **Exact Match:** Verifica dell'esistenza esatta della tripla nel grafo.
2.  **Fallback Semantico:** Qualora non esista un match letterale tra i predicati, il sistema deve calcolare la similarità del coseno tra l'embedding del predicato del claim e i predicati presenti tra i due nodi nel grafo.
3.  **Validazione:** Se la similarità supera una soglia definita, l'attribuzione è confermata e la fonte viene estratta direttamente dalle proprietà dell'arco.

### Oggetto di studio A: Confronto nella coppia <claim, context>

Un'LLM genererà la risposta partendo dai passaggi forniti da ALCE, il context dovrebbe essere già embeddato in un grafo (runTime o già presente nel DB) e i claim della risposta verranno estratti con la stessa tecnologia (DeepSeek) utilizzata per l'ingestione — la simmetria di estrazione è il requisito §4.

La pipeline che si dovrebbe andare a creare per lo studio della coppia <claim, context> è la seguente:
1. **Generazione del Generator con i passaggi forniti da ASQA** LLM genera la risposta assumendo di aver fornito i passaggi giusti dal corpus ALCE
2. **Claim Extraction** I claim vengono estratti dalla riposta generata (con DeepSeek, lo stesso estrattore dell'ingestione)
3. **Claim Attribution** Verrà effettuata una ricerca all'interno del Context fornito per recuperare i passaggi pertinenti ai claim estratti (se esiste il predicato che unisce i due nodi allora il claim è supportato, altrimenti si controlla il feedback semantico, se anche questo non porta a nulla allora si scarta il claim).
4. **Generazione della Risposta finale** Sulla base dei claim supportati, verrà generata la risposta finale.

## 3. SPECIFICHE DELL'INTERFACCIA UTENTE (UI)
* **Tecnologia:** Streamlit o React (scelta basata sull'ottimizzazione del consumo di token e velocità di sviluppo).
* **Funzionalità Core:**
    * Visualizzazione dello stato di elaborazione (Cleaning -> Extraction -> Indexing).
    * Modulo di test per la "Claim Attribution": l'utente inserisce un'affermazione e il sistema restituisce la validazione, il grado di similarità e il chunk di testo sorgente.

### Esperimenti nella UI
Gli esperimenti (pipeline ibrida REBEL+DeepSeek, varianti A/D, statistiche
della cache LLM, runner batch) sono **strumenti di sviluppo, non funzionalita'
del sistema**: vivono in `src/ui/experiments.py` e il loro tab compare solo con
`SHOW_EXPERIMENTS=1` nell'ambiente.  Senza la variabile l'app espone i soli tab
**Corpus ALCE** e **Claim Attribution**.  Le risorse Streamlit condivise stanno
in `src/ui/resources.py`: un `@st.cache_resource` duplicato in due moduli
caricherebbe i modelli due volte.

## 4. REGOLE PER L'AGENTE (CLAUDE CODE)
* **Priorità:** La coerenza tra la fase di encoding del corpus e la fase di processing della risposta deve essere assoluta (stesso modello di estrazione).
* **Documentazione:** Ogni modifica al database o alla logica di matching deve essere documentata nel file `tasks/lessons.md`.
* **Simplicità:** Prediligere librerie specializzate (es. NetworkX per prototipazione veloce, driver ufficiale Neo4j per produzione) evitando l'over-engineering di modelli generativi dove bastano algoritmi deterministici.
