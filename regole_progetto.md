## 1. OBIETTIVO E SCOPO DEL PROGETTO
L'obiettivo è la realizzazione di un sistema di "Claim Attribution" oggettivo e deterministico. Il sistema deve superare i limiti dei RAG vettoriali puri utilizzando un Labeled Property Graph (LPG) come fonte di verità. 
**Nota di Override:** Questo progetto sostituisce integralmente ogni precedente iterazione riguardante pipeline "in-generation". Il focus esclusivo è la trasformazione del corpus in triple e la successiva validazione dei claim generati.

## 2. ARCHITETTURA DEL SISTEMA E DATA PIPELINE

### Fase 1: Ingestione e Pre-processing del Corpus
Ogni file caricato (PDF o TXT) deve essere processato secondo la seguente catena:
1.  **Semantic Cleaning (LLM-Based):** Utilizzare un modello linguistico (Llama-3 o superiore) esclusivamente per la risoluzione delle coreferenze. Ogni pronome deve essere sostituito con il soggetto esplicito a cui si riferisce per massimizzare la qualità delle triple estratte, stesse entità vanno accorpate semanticamente in un unico soggetto per il contesto (es: Cristiano Ronaldo, CR7, Ronaldo vanno accorpati in Cristiano Ronaldo).
2.  **Estrazione delle Relazioni:** Trasformazione del testo pulito in triple RDFS/LPG `(Soggetto, Predicato, Oggetto)` tramite modello specializzato (es. REBEL).
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

## 3. SPECIFICHE DELL'INTERFACCIA UTENTE (UI)
* **Tecnologia:** Streamlit o React (scelta basata sull'ottimizzazione del consumo di token e velocità di sviluppo).
* **Funzionalità Core:**
    * Area di upload per file PDF/TXT.
    * Visualizzazione dello stato di elaborazione (Cleaning -> Extraction -> Indexing).
    * Modulo di test per la "Claim Attribution": l'utente inserisce un'affermazione e il sistema restituisce la validazione, il grado di similarità e il chunk di testo sorgente.

## 4. REGOLE PER L'AGENTE (CLAUDE CODE)
* **Priorità:** La coerenza tra la fase di encoding del corpus e la fase di processing della risposta deve essere assoluta (stesso modello di estrazione).
* **Documentazione:** Ogni modifica al database o alla logica di matching deve essere documentata nel file `tasks/lessons.md`.
* **Simplicità:** Prediligere librerie specializzate (es. NetworkX per prototipazione veloce, driver ufficiale Neo4j per produzione) evitando l'over-engineering di modelli generativi dove bastano algoritmi deterministici.
