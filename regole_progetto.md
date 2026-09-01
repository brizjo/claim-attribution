## 1. OBIETTIVO E SCOPO DEL PROGETTO
L'obiettivo è la realizzazione di un sistema di "Claim Attribution" oggettivo e deterministico. Il sistema deve superare i limiti dei RAG vettoriali puri utilizzando un Labeled Property Graph (LPG) come fonte di verità. 


## 2. ARCHITETTURA DEL SISTEMA E DATA PIPELINE

### Fase 1: Ingestione e Pre-processing del Corpus
Non sono caricati file dall'utente, il corpus è preso da un dataset già esistente (ASQA). Esso deve essere preprocessato, attraverso la divisione in chunk, la risoluzione delle coreferenze e  infine l'estrazione delle triple, tramite REBEL, previa una divisione del passaggio in frasi semplici attraverso un sentence tokenizer. Considerando che l'output di REBEL ha un vocabolario chiuso e ben definito ma non riesce a coprire tutte le possibili relazioni, è necessario che il modello linguistico di DeepSeek estragga, a partire dal chunk e dall'output di REBEL, un set completo di triple, coprendo ciò che REBEL non riesce ad estrarre.

3.  **Mapping e Metadati:** Ogni relazione creata nel database deve includere come proprietà:
    * Il testo del chunk originale in chiaro.
    * Il riferimento al documento (nome file, autore, data).
    * L'embedding vettoriale della stringa del predicato.

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

Un'LLM genererà la risposta partendo dai passaggi forniti da ALCE, il context dovrebbe essere già embeddato in un grafo (runTime o già presente nel DB) e i claim della risposta verranno estratti con la stessa tecnologia (REBEL+DeepSeek) utilizzata per l'ingestione.

La pipeline che si dovrebbe andare a creare per lo studio della coppia <claim, context> è la seguente:
1. **Generazione del Generator con i passaggi forniti da ASQA** LLM genera la risposta assumendo di aver fornito i passaggi giusti dal corpus ALCE
2. **Claim Extraction** I claim vengono estratti dalla riposta generata (con REBEL + DeepSeek)
3. **Claim Attribution** Verrà effettuata una ricerca all'interno del Context fornito per recuperare i passaggi pertinenti ai claim estratti (se esiste il predicato che unisce i due nodi allora il claim è supportato, altrimenti si controlla il feedback semantico, se anche questo non porta a nulla allora si scarta il claim).
4. **Generazione della Risposta finale** Sulla base dei claim supportati, verrà generata la risposta finale.

## 3. SPECIFICHE DELL'INTERFACCIA UTENTE (UI)
* **Tecnologia:** Streamlit o React (scelta basata sull'ottimizzazione del consumo di token e velocità di sviluppo).
* **Funzionalità Core:**
    * Visualizzazione dello stato di elaborazione (Cleaning -> Extraction -> Indexing).
    * Modulo di test per la "Claim Attribution": l'utente inserisce un'affermazione e il sistema restituisce la validazione, il grado di similarità e il chunk di testo sorgente.

## 4. REGOLE PER L'AGENTE (CLAUDE CODE)
* **Priorità:** La coerenza tra la fase di encoding del corpus e la fase di processing della risposta deve essere assoluta (stesso modello di estrazione).
* **Documentazione:** Ogni modifica al database o alla logica di matching deve essere documentata nel file `tasks/lessons.md`.
* **Simplicità:** Prediligere librerie specializzate (es. NetworkX per prototipazione veloce, driver ufficiale Neo4j per produzione) evitando l'over-engineering di modelli generativi dove bastano algoritmi deterministici.
