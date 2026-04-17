"""
Test suite per la valutazione del sistema RAG su RAG-RewardBench.

===================================================================
SCOPO:
Questo file è predisposto per caricare e valutare il sistema di
claim attribution sul benchmark RAG-RewardBench.

RAG-RewardBench è un benchmark progettato per valutare la capacità
dei sistemi RAG di gestire CONFLITTI TRA LE FONTI. Testa se il
sistema è in grado di:

1. Rilevare quando fonti diverse forniscono informazioni conflittuali
2. Assegnare correttamente gli score di attribuzione in presenza di
   documenti contraddittori
3. Distinguere tra risposte ben supportate e allucinazioni quando
   il contesto è ambiguo o conflittuale

Il benchmark valuta la ROBUSTEZZA del sistema nei seguenti scenari:
- Fonti concordanti (tutti i documenti supportano la stessa risposta)
- Fonti parzialmente conflittuali (alcune fonti concordano, altre no)
- Fonti completamente conflittuali (le fonti si contraddicono a vicenda)
- Fonti irrilevanti (i documenti recuperati non riguardano la domanda)

RIFERIMENTI:
- RAG-RewardBench: https://huggingface.co/datasets/...
- Paper di riferimento: da specificare
===================================================================

TODO:
- [ ] Integrare il caricamento del dataset RAG-RewardBench da HuggingFace
- [ ] Implementare le metriche di valutazione specifiche del benchmark
- [ ] Aggiungere test per scenari di conflitto tra fonti
- [ ] Confrontare le performance del sistema con le baseline del paper
"""

import sys
import os
from pathlib import Path

import pytest
import numpy as np

# Aggiungi la root del progetto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.sentence_splitter import SentenceSplitter
from src.attribution.lexical_overlap import LexicalOverlap
from src.attribution.semantic_similarity import SemanticSimilarity
from src.attribution.matrix import SupportMatrix


# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture
def splitter():
    """Inizializza il sentence splitter per i test."""
    return SentenceSplitter()


@pytest.fixture
def lexical():
    """Inizializza il modulo di overlap lessicale."""
    return LexicalOverlap()


@pytest.fixture
def support_matrix():
    """Inizializza la matrice di supporto."""
    return SupportMatrix()


@pytest.fixture
def sample_concordant_context():
    """
    Scenario: fonti CONCORDANTI.
    Tutti i documenti supportano la stessa informazione.
    Il sistema dovrebbe assegnare score alti.
    """
    return {
        "question": "What is the capital of France?",
        "documents": [
            {
                "document": "Paris is the capital and largest city of France. "
                            "It is situated on the River Seine.",
                "metadata": {"source": "Wikipedia - Paris", "chunk_index": 0},
            },
            {
                "document": "The capital of France is Paris, which has been the "
                            "country's capital since the 10th century.",
                "metadata": {"source": "Wikipedia - France", "chunk_index": 0},
            },
        ],
        "expected_response": "The capital of France is Paris.",
        "expected_min_score": 0.7,  # Alto supporto atteso
    }


@pytest.fixture
def sample_conflicting_context():
    """
    Scenario: fonti CONFLITTUALI.
    I documenti forniscono informazioni contraddittorie.
    Il sistema dovrebbe rilevare il conflitto e assegnare score
    appropriati (non alti per tutti).

    NOTA: Questo scenario testa la robustezza ai conflitti tra fonti,
    che è il focus principale di RAG-RewardBench.
    """
    return {
        "question": "What is the population of the city?",
        "documents": [
            {
                "document": "The city has a population of approximately 2.1 million "
                            "people according to the 2020 census.",
                "metadata": {"source": "Source A", "chunk_index": 0},
            },
            {
                "document": "The population of the city is estimated at 8.3 million "
                            "inhabitants as of the latest survey.",
                "metadata": {"source": "Source B", "chunk_index": 0},
            },
        ],
        "expected_response": "The city has a population of approximately 2.1 million people.",
        # Solo una fonte supporta questa risposta; l'altra la contraddice
    }


@pytest.fixture
def sample_irrelevant_context():
    """
    Scenario: fonti IRRILEVANTI.
    I documenti recuperati non riguardano la domanda.
    Il sistema dovrebbe assegnare score bassi (potenziale allucinazione).
    """
    return {
        "question": "What causes earthquakes?",
        "documents": [
            {
                "document": "Chocolate is made from cacao beans that are harvested "
                            "from tropical regions near the equator.",
                "metadata": {"source": "Wikipedia - Chocolate", "chunk_index": 0},
            },
            {
                "document": "The Eiffel Tower was built in 1889 for the World's Fair "
                            "and stands 330 meters tall.",
                "metadata": {"source": "Wikipedia - Eiffel Tower", "chunk_index": 0},
            },
        ],
        "expected_response": "Earthquakes are caused by the movement of tectonic plates.",
        "expected_max_score": 0.3,  # Basso supporto atteso (non supportato)
    }


# ====================================================================
# Unit Tests — Componenti base
# ====================================================================


class TestSentenceSplitter:
    """Test per la segmentazione a livello di frase."""

    def test_basic_split(self, splitter):
        """Verifica che il testo venga correttamente diviso in frasi."""
        text = "This is the first sentence. This is the second sentence."
        sentences = splitter.split(text)
        assert len(sentences) >= 2
        assert any("first" in s for s in sentences)

    def test_atomic_fact_extraction(self, splitter):
        """Verifica l'estrazione di fatti atomici dalla risposta."""
        response = "Paris is the capital of France. It has a population of 2 million. The Eiffel Tower is located there."
        facts = splitter.extract_atomic_facts(response)
        assert len(facts) >= 2
        # Ogni fatto dovrebbe avere almeno 3 parole
        for fact in facts:
            assert len(fact.split()) >= 3

    def test_empty_input(self, splitter):
        """Verifica il comportamento con input vuoto."""
        assert splitter.split("") == []
        assert splitter.extract_atomic_facts("") == []


class TestLexicalOverlap:
    """Test per le metriche di overlap lessicale."""

    def test_exact_match_identical(self, lexical):
        """Match esatto per testi identici."""
        score = lexical.exact_match("hello world", "hello world")
        assert score == 1.0

    def test_exact_match_containment(self, lexical):
        """Contenenza parziale."""
        score = lexical.exact_match("hello", "hello world")
        assert score > 0.0

    def test_exact_match_no_overlap(self, lexical):
        """Nessuna sovrapposizione."""
        score = lexical.exact_match("foo bar", "baz qux")
        assert score == 0.0

    def test_rouge_l_similar(self, lexical):
        """ROUGE-L per testi simili dovrebbe essere alto."""
        score = lexical.rouge_l(
            "The capital of France is Paris",
            "Paris is the capital of France"
        )
        assert score > 0.5

    def test_rouge_l_different(self, lexical):
        """ROUGE-L per testi diversi dovrebbe essere basso."""
        score = lexical.rouge_l(
            "The weather is sunny today",
            "Quantum physics describes subatomic particles"
        )
        assert score < 0.3


# ====================================================================
# Integration Tests — Scenari RAG-RewardBench
# ====================================================================


class TestConcordantSources:
    """
    Test per scenari con fonti concordanti.

    RAG-RewardBench scenario: tutte le fonti supportano la risposta.
    Il sistema dovrebbe produrre score di attribuzione ALTI.
    """

    def test_high_support_with_concordant_sources(
        self, splitter, support_matrix, sample_concordant_context
    ):
        """
        Verifica che fonti concordanti producano score alti.

        Questo è il caso ideale per un sistema RAG: i documenti
        recuperati concordano e supportano la risposta generata.
        """
        ctx = sample_concordant_context

        # Segmenta i documenti in frasi
        evidence = splitter.split_documents(ctx["documents"])
        assert len(evidence) > 0, "Nessuna evidenza estratta dai documenti"

        # Estrai fatti atomici dalla risposta
        facts = splitter.extract_atomic_facts(ctx["expected_response"])
        assert len(facts) > 0, "Nessun fatto atomico estratto"

        # Calcola la matrice di supporto
        # NOTA: Questo test usa solo lexical overlap per velocità;
        #       in produzione si usa anche BERTScore.
        lexical = LexicalOverlap()
        lex_matrix = lexical.score_matrix(
            facts, [ev["sentence"] for ev in evidence]
        )

        # Ogni fatto dovrebbe avere almeno un'evidenza con score > 0.3
        row_maxes = np.max(lex_matrix, axis=1)
        for i, max_score in enumerate(row_maxes):
            assert max_score > 0.1, (
                f"Il fatto '{facts[i]}' non ha supporto lessicale sufficiente "
                f"nonostante le fonti concordanti (max_score={max_score:.3f})"
            )


class TestConflictingSources:
    """
    Test per scenari con fonti CONFLITTUALI.

    RAG-RewardBench scenario: le fonti si contraddicono a vicenda.
    Il sistema dovrebbe:
    - Assegnare score alti solo per la fonte che supporta la risposta
    - Non assegnare score alti per la fonte conflittuale

    NOTA: Questo è il test più critico per la robustezza del sistema.
    Un sistema robusto ai conflitti tra fonti dovrebbe essere in grado
    di distinguere quale fonte supporta effettivamente la risposta.
    """

    def test_selective_attribution_with_conflicts(
        self, splitter, sample_conflicting_context
    ):
        """
        Verifica che il sistema attribuisca correttamente quando
        le fonti sono in conflitto.
        """
        ctx = sample_conflicting_context

        evidence = splitter.split_documents(ctx["documents"])
        facts = splitter.extract_atomic_facts(ctx["expected_response"])

        lexical = LexicalOverlap()
        lex_matrix = lexical.score_matrix(
            facts, [ev["sentence"] for ev in evidence]
        )

        # La risposta menziona "2.1 million" → dovrebbe matchare Source A
        # ma NON Source B (che dice "8.3 million")
        # Verifica che la matrice non sia uniformemente alta
        if lex_matrix.size > 0:
            col_maxes = np.max(lex_matrix, axis=0)
            # Non tutti i contesti dovrebbero avere score alto
            # (almeno una colonna dovrebbe avere score basso)
            assert np.min(col_maxes) < np.max(col_maxes), (
                "Score uniformi nonostante fonti conflittuali — "
                "il sistema non distingue le fonti"
            )


class TestIrrelevantSources:
    """
    Test per scenari con fonti IRRILEVANTI.

    RAG-RewardBench scenario: i documenti recuperati non c'entrano
    con la domanda. Il sistema dovrebbe assegnare score BASSI,
    segnalando un alto rischio di allucinazione.
    """

    def test_low_support_with_irrelevant_sources(
        self, splitter, sample_irrelevant_context
    ):
        """
        Verifica che fonti irrilevanti producano score bassi.

        Se i documenti non sono pertinenti, la risposta è necessariamente
        un'allucinazione e gli score di attribuzione devono rifletterlo.
        """
        ctx = sample_irrelevant_context

        evidence = splitter.split_documents(ctx["documents"])
        facts = splitter.extract_atomic_facts(ctx["expected_response"])

        lexical = LexicalOverlap()
        lex_matrix = lexical.score_matrix(
            facts, [ev["sentence"] for ev in evidence]
        )

        # Tutti gli score dovrebbero essere bassi
        if lex_matrix.size > 0:
            overall_max = np.max(lex_matrix)
            assert overall_max < ctx["expected_max_score"], (
                f"Score troppo alto ({overall_max:.3f}) nonostante "
                f"fonti completamente irrilevanti"
            )


# ====================================================================
# Benchmark Loading (Predisposizione per RAG-RewardBench)
# ====================================================================


class TestRAGRewardBenchIntegration:
    """
    Predisposizione per il caricamento e la valutazione completa
    sul benchmark RAG-RewardBench.

    TODO: Implementare quando il dataset sarà disponibile.

    Il benchmark testa la robustezza del sistema ai conflitti tra
    le fonti con i seguenti task:
    - Information Retrieval quality
    - Source conflict detection
    - Attribution accuracy under contradiction
    - Hallucination detection in ambiguous contexts
    """

    @pytest.mark.skip(reason="RAG-RewardBench dataset non ancora configurato")
    def test_load_rag_rewardbench_dataset(self):
        """
        Carica il dataset RAG-RewardBench da HuggingFace.

        Utilizzo previsto:
            from datasets import load_dataset
            dataset = load_dataset("rag-rewardbench", split="test")

        Il dataset dovrebbe contenere:
        - query: la domanda originale
        - documents: lista di documenti di contesto (potenzialmente conflittuali)
        - reference_answer: la risposta corretta di riferimento
        - conflict_type: tipo di conflitto (concordant/partial/full/irrelevant)
        """
        from datasets import load_dataset

        # TODO: Sostituire con il nome corretto del dataset
        # dataset = load_dataset("rag-rewardbench", split="test")
        # assert len(dataset) > 0
        pass

    @pytest.mark.skip(reason="RAG-RewardBench evaluation non ancora implementata")
    def test_evaluate_on_rag_rewardbench(self):
        """
        Esegue la valutazione completa sul benchmark.

        Metriche da calcolare:
        - Attribution Accuracy: % di fatti correttamente attribuiti
        - Conflict Detection Rate: % di conflitti rilevati
        - Hallucination False Positive Rate: % di allucinazioni non rilevate
        - Source Discrimination Score: capacità di distinguere fonti affidabili

        Questo test serve per confrontare le performance del nostro sistema
        con le baseline riportate nel paper di RAG-RewardBench e verificare
        la robustezza ai conflitti tra le fonti.
        """
        # TODO: Implementare il loop di valutazione
        #
        # for sample in dataset:
        #     query = sample["query"]
        #     documents = sample["documents"]
        #     reference = sample["reference_answer"]
        #     conflict_type = sample["conflict_type"]
        #
        #     # 1. Segmenta documenti
        #     evidence = splitter.split_documents(documents)
        #
        #     # 2. Genera risposta (o usa la reference per test)
        #     response = reference
        #
        #     # 3. Estrai fatti atomici
        #     facts = splitter.extract_atomic_facts(response)
        #
        #     # 4. Calcola matrice di supporto
        #     matrix.compute(facts, evidence)
        #     results = matrix.get_attribution_results()
        #
        #     # 5. Confronta con ground truth del benchmark
        #     ...
        pass
