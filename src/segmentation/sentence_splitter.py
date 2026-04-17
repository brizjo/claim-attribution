"""
Modulo Segmentation — Frammentazione a livello di singola frase.

Asse X della matrice di supporto: prende i chunk di contesto recuperati
e li scompone in singole frasi (sentence-level evidence) usando spaCy.
"""

from typing import Optional

from config import settings


class SentenceSplitter:
    """
    Segmenta testo in singole frasi utilizzando spaCy.

    Questa classe implementa la frammentazione sentence-level necessaria
    per costruire l'asse X (contesti) della matrice di supporto.
    """

    def __init__(self, spacy_model: str = settings.SPACY_MODEL):
        """
        Inizializza il sentence splitter caricando il modello spaCy.

        Args:
            spacy_model: Nome del modello spaCy da utilizzare.
        """
        try:
            import spacy
            self.nlp = spacy.load(spacy_model)
        except OSError:
            # Fallback: se il modello non è installato, usa il sentencizer base
            import spacy
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")

    # ────────────────────────────────────────────────────────────────
    # Core: split di un singolo testo
    # ────────────────────────────────────────────────────────────────

    def split(self, text: str) -> list[str]:
        """
        Divide un testo in una lista di frasi.

        Args:
            text: Testo da segmentare.

        Returns:
            Lista di frasi (stringhe).
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences

    # ────────────────────────────────────────────────────────────────
    # Split di più chunk di contesto
    # ────────────────────────────────────────────────────────────────

    def split_documents(
        self, documents: list[dict]
    ) -> list[dict[str, str]]:
        """
        Prende una lista di documenti dal retriever e li frammenta
        in singole frasi, preservando i metadati di provenienza.

        Args:
            documents: Lista di dict con chiavi "document" e "metadata".

        Returns:
            Lista di dict con chiavi:
              - "sentence": la frase singola
              - "source":   titolo della fonte originale
              - "chunk_index": indice del chunk di provenienza
        """
        evidence_sentences = []

        for doc in documents:
            text = doc.get("document", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            chunk_idx = metadata.get("chunk_index", -1)

            sentences = self.split(text)
            for sentence in sentences:
                evidence_sentences.append(
                    {
                        "sentence": sentence,
                        "source": source,
                        "chunk_index": chunk_idx,
                    }
                )

        return evidence_sentences

    # ────────────────────────────────────────────────────────────────
    # Estrazione fatti atomici dalla risposta (Mock / Placeholder)
    # ────────────────────────────────────────────────────────────────

    def extract_atomic_facts(self, generated_response: str) -> list[str]:
        """
        Estrae i fatti atomici dalla risposta generata dall'LLM.

        NOTA: Questa è una implementazione placeholder/mock.
        In produzione, questa funzione dovrebbe utilizzare un modello
        NLI o un LLM dedicato per decomporre la risposta in claim
        atomiche verificabili (es. FActScore, SAFE decomposition).

        Per ora, utilizziamo la segmentazione a livello di frase come
        approssimazione dei fatti atomici.

        Args:
            generated_response: La risposta generata dall'LLM.

        Returns:
            Lista di fatti atomici (stringhe).
        """
        # TODO: Sostituire con decomposizione atomica vera
        # (es. usando un LLM per estrarre claim indipendenti)
        #
        # Esempio di approccio futuro:
        #   prompt = f"Decompose the following text into atomic facts:\n{generated_response}"
        #   atomic_facts = llm.generate(prompt)
        #
        # Per ora: ogni frase della risposta = un fatto atomico
        atomic_facts = self.split(generated_response)

        # Filtra frasi troppo corte o non informative
        atomic_facts = [
            fact for fact in atomic_facts
            if len(fact.split()) >= 3  # almeno 3 parole
        ]

        return atomic_facts
