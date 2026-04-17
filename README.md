# RAG Claim Attribution System

> **Progetto di ricerca universitario** — Sistema di Retrieval-Augmented Generation con claim attribution basato su highlight gradient.

## Panoramica

Questo sistema implementa un approccio **Post-Retrieval** alla claim attribution per modelli RAG. Data una query dell'utente:

1. **Retrieval**: vengono recuperati documenti rilevanti da Wikipedia (database vettoriale)
2. **Generation**: Llama-3 (via Ollama) genera una risposta basata sui documenti recuperati
3. **Segmentation**: i documenti recuperati vengono frammentati a livello di singola frase (sentence-level evidence)
4. **Attribution Matrix**: ogni fatto atomico della risposta viene incrociato con le frasi del contesto, calcolando score di supporto tramite metriche non-generative (BERTScore, ROUGE, Exact Match)
5. **Visualization**: il testo generato viene colorato con un **highlight gradient** (verde → rosso) in base al livello di supporto

## Architettura

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  User Query  │────▶│  Wikipedia   │────▶│  Llama-3 (Ollama)│
│  (Streamlit) │     │  Retriever   │     │  Generator       │
└─────────────┘     └──────────────┘     └────────┬─────────┘
                           │                       │
                    ┌──────▼──────┐         ┌──────▼─────────┐
                    │  Sentence   │         │  Atomic Fact    │
                    │  Splitter   │         │  Extractor      │
                    └──────┬──────┘         └──────┬─────────┘
                           │                       │
                           └───────────┬───────────┘
                                ┌──────▼──────────┐
                                │  Support Matrix  │
                                │  (BERTScore +    │
                                │   Lexical Overlap│)
                                └──────┬──────────┘
                                       │
                                ┌──────▼──────────┐
                                │  Highlight       │
                                │  Gradient UI     │
                                └─────────────────┘
```

## Stack Tecnologico

| Componente        | Tecnologia                          |
|-------------------|-------------------------------------|
| Frontend / UI     | Streamlit                           |
| LLM Locale        | Llama-3 (via Ollama)                |
| Knowledge Base    | Wikipedia (ChromaDB vector store)   |
| Embeddings        | SentenceTransformers                |
| Similarità Sem.   | BERTScore                           |
| Overlap Lessicale | ROUGE-L, Exact Match                |
| Segmentazione     | spaCy / NLTK                        |

## Struttura del Progetto

```
rag-claim-attribution/
├── app.py                         # Interfaccia Streamlit principale
├── requirements.txt               # Dipendenze Python
├── README.md                      # Questo file
├── .gitignore                     # Git ignore
├── config/
│   └── settings.py                # Configurazione centralizzata
├── src/
│   ├── __init__.py
│   ├── retriever/
│   │   ├── __init__.py
│   │   └── wiki_retriever.py      # Recupero documenti da Wikipedia
│   ├── generator/
│   │   ├── __init__.py
│   │   └── llama_generator.py     # Generazione con Llama-3 via Ollama
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── sentence_splitter.py   # Frammentazione a livello di frase
│   ├── attribution/
│   │   ├── __init__.py
│   │   ├── matrix.py              # Matrice di supporto output-contesti
│   │   ├── semantic_similarity.py # BERTScore
│   │   └── lexical_overlap.py     # ROUGE-L e Exact Match
│   └── visualization/
│       ├── __init__.py
│       └── highlight_renderer.py  # Rendering highlight gradient
├── tests/
│   ├── __init__.py
│   └── test_rag_rewardbench.py    # Valutazione su RAG-RewardBench
└── data/
    └── .gitkeep
```

## Setup e Installazione

### Prerequisiti

- Python 3.10+
- [Ollama](https://ollama.ai/) installato con il modello Llama-3 disponibile

### Installazione

```bash
# Clona il repository
git clone https://github.com/<your-username>/rag-claim-attribution.git
cd rag-claim-attribution

# Crea un virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installa le dipendenze
pip install -r requirements.txt

# Scarica il modello spaCy per la segmentazione
python -m spacy download en_core_web_sm

# Assicurati che Ollama sia attivo e che Llama-3 sia disponibile
ollama pull llama3
```

### Esecuzione

```bash
streamlit run app.py
```

## Utilizzo

1. Inserisci una query nella barra di input
2. Il sistema recupera i documenti rilevanti da Wikipedia
3. Llama-3 genera una risposta contestualizzata
4. La risposta viene visualizzata con **highlight gradient**:
   - 🟢 **Verde intenso**: alto supporto dalle fonti (score ≥ 0.8)
   - 🟡 **Giallo**: supporto parziale (score 0.5–0.8)
   - 🟠 **Arancione**: supporto debole (score 0.3–0.5)
   - 🔴 **Rosso**: potenziale allucinazione (score < 0.3)

## Valutazione

Il sistema è predisposto per la valutazione sul benchmark **RAG-RewardBench** per testare la robustezza ai conflitti tra fonti:

```bash
pytest tests/test_rag_rewardbench.py -v
```

## Riferimenti

- Gao et al., *"Enabling Large Language Models to Generate Text with Citations"*, 2023
- Es et al., *"RAGAs: Automated Evaluation of Retrieval Augmented Generation"*, 2024
- RAG-RewardBench: benchmark per la valutazione di sistemi RAG

## Licenza

Questo progetto è sviluppato per scopi di ricerca universitaria.
