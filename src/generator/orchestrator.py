"""
Modulo Orchestrator — Cuore del pipeline In-Generation Attribution.

Implementa il loop "Chain-of-Citation":
  1. Prompt iniziale con istruzioni per usare <CERCA: [query]>
  2. L'LLM genera fino a emettere <CERCA:> → HALT
  3. Vector search con la query estratta
  4. Iniezione del contesto recuperato nel prompt
  5. Riprende la generazione (loop fino a MAX_CERCA_ITERATIONS)
  6. Raffinamento del testo intermedio in risposta finale

Supporta due modalità:
  - Automatic: la retrieval avviene immediatamente
  - Interactive (Demonstration): si ferma e aspetta un click dell'utente
"""

from dataclasses import dataclass, field
from typing import Optional, Callable

from config import settings
from src.generator.llama_generator import LlamaGenerator
from src.retriever.vector_retriever import VectorRetriever


@dataclass
class CercaEvent:
    """
    Rappresenta un singolo evento CERCA (stop-retrieve-resume).

    Attributes:
        iteration:       Numero dell'iterazione (1-based).
        cerca_query:     La query di ricerca estratta dal tag <CERCA:>.
        partial_text:    Il testo generato PRIMA del tag <CERCA:>.
        retrieved_chunks: I chunk recuperati dal vector DB.
        resumed:         True se la generazione è stata ripresa dopo questo evento.
    """
    iteration: int
    cerca_query: str
    partial_text: str
    retrieved_chunks: list[dict] = field(default_factory=list)
    resumed: bool = False


@dataclass
class OrchestrationResult:
    """
    Risultato completo dell'orchestrazione In-Generation.

    Attributes:
        question:          La domanda originale dell'utente.
        intermediate_text: Il testo "grezzo" generato con tutte le iterazioni.
        final_response:    Il testo raffinato (pulito e fluido).
        cerca_events:      Lista di tutti gli eventi CERCA (provenance log).
        all_sources:       Tutti i chunk recuperati (con source_id e testo).
        total_iterations:  Numero totale di iterazioni CERCA effettuate.
    """
    question: str
    intermediate_text: str = ""
    final_response: str = ""
    cerca_events: list[CercaEvent] = field(default_factory=list)
    all_sources: list[dict] = field(default_factory=list)
    total_iterations: int = 0


class InGenerationOrchestrator:
    """
    Orchestratore per il pipeline In-Generation Attribution.

    Gestisce il loop completo:
      prompt → generate → detect <CERCA:> → halt → retrieve → inject → resume
    """

    def __init__(
        self,
        generator: LlamaGenerator,
        retriever: VectorRetriever,
        max_iterations: int = settings.MAX_CERCA_ITERATIONS,
        top_k: int = settings.TOP_K_DOCUMENTS,
    ):
        self.generator = generator
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.top_k = top_k

    # ────────────────────────────────────────────────────────────────
    # Prompt Building (Using Llama-3 raw tokens for Assistant Prefilling)
    # ────────────────────────────────────────────────────────────────

    def _build_initial_prompt(self, question: str) -> str:
        """Costruisce il prompt iniziale con istruzioni Chain-of-Citation."""
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{settings.CHAIN_OF_CITATION_SYSTEM}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt

    def _build_resume_prompt(
        self,
        question: str,
        partial_text: str,
        all_sources: list[dict],
        past_queries: list[str] = None,
    ) -> str:
        """
        Costruisce il prompt per riprendere la generazione simulando un uso dello strumento.
        """
        if past_queries is None:
            past_queries = []
            
        latest_query = past_queries[-1] if past_queries else "unknown"

        # Formatta le fonti
        source_parts = []
        for i, source in enumerate(all_sources, start=1):
            src_name = source.get("metadata", {}).get("source", "Unknown")
            src_text = source.get("document", "")
            source_parts.append(f"[{i}] (Source: {src_name})\n{src_text}")
        sources_str = "\n\n".join(source_parts)

        # Costruisce la history multi-turn
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{settings.CHAIN_OF_CITATION_SYSTEM}<|eot_id|>"
            
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|>"
            
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{partial_text} <CERCA: {latest_query}><|eot_id|>"
            
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"DATABASE RESULTS FOR YOUR SEARCH:\n{sources_str}\n\n"
            f"Instructions:\n"
            f"1. You have successfully retrieved the sources. DO NOT search for this again.\n"
            f"2. Seamlessly continue the last sentence from your previous turn, incorporating the new facts.\n"
            f"3. Cite your facts using [1], [2].\n"
            f"4. If you need a DIFFERENT fact, you may output a new <CERCA: new_keyword>.\n"
            f"5. Do not output introductory conversational fillers.<|eot_id|>"
            
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt

    # ────────────────────────────────────────────────────────────────
    # Orchestrazione Step-by-Step (per modalità interattiva)
    # ────────────────────────────────────────────────────────────────

    def start_generation(
        self,
        question: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        Avvia la generazione iniziale. Ritorna quando:
          - Il modello emette <CERCA:> (stopped=True)
          - Il modello completa la risposta (stopped=False)

        Returns:
            Dict con chiavi: text, cerca_query, stopped, full_raw_text
        """
        prompt = self._build_initial_prompt(question)
        return self.generator.generate_with_stop(
            prompt, stream_callback=stream_callback
        )

    def resume_after_cerca(
        self,
        question: str,
        partial_text: str,
        all_sources: list[dict],
        past_queries: list[str] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        Riprende la generazione dopo un evento CERCA con nuovo contesto.

        Args:
            question:     La domanda originale.
            partial_text: Tutto il testo generato finora.
            all_sources:  Tutti i chunk recuperati finora.
            past_queries: Query CERCA precedenti (per evitare ripetizioni).

        Returns:
            Dict con chiavi: text, cerca_query, stopped, full_raw_text
        """
        prompt = self._build_resume_prompt(
            question, partial_text, all_sources, past_queries
        )
        return self.generator.generate_with_stop(
            prompt, stream_callback=stream_callback
        )

    # ────────────────────────────────────────────────────────────────
    # Orchestrazione Automatica (loop completo)
    # ────────────────────────────────────────────────────────────────

    def run_automatic(
        self,
        question: str,
        status_callback: Optional[Callable[[str], None]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> OrchestrationResult:
        """
        Esegue il loop completo In-Generation Attribution in modalità automatica.

        Il loop:
          1. Genera con Chain-of-Citation prompt
          2. Se <CERCA:> → halt, retrieve, inject, retry (fino a max_iterations)
          3. Quando il modello completa → refine → return

        Args:
            question:        La domanda dell'utente.
            status_callback: Callback per aggiornare lo stato nell'UI.

        Returns:
            OrchestrationResult con testo intermedio, finale, e provenance.
        """
        result = OrchestrationResult(question=question)
        all_sources: list[dict] = []
        accumulated_text = ""
        iteration = 0

        if status_callback:
            status_callback("🤖 Avvio generazione con Chain-of-Citation...")

        # ── Step 1: Generazione iniziale ──────────────────────────
        gen_result = self.start_generation(
            question, stream_callback=stream_callback
        )
        accumulated_text = gen_result["text"]

        # ── Step 2: Loop CERCA ────────────────────────────────────
        while gen_result["stopped"] and iteration < self.max_iterations:
            iteration += 1
            cerca_query = gen_result["cerca_query"]

            if status_callback:
                status_callback(
                    f"🔍 CERCA #{iteration}: \"{cerca_query}\" — Ricerca in corso..."
                )

            # Cerca nel vector DB
            retrieved = self.retriever.query(cerca_query, top_k=self.top_k)

            # Registra l'evento CERCA con provenance
            event = CercaEvent(
                iteration=iteration,
                cerca_query=cerca_query,
                partial_text=accumulated_text,
                retrieved_chunks=retrieved,
            )

            # Accumula tutte le fonti (con deduplicazione per source_id)
            existing_ids = {s.get("source_id") for s in all_sources}
            for chunk in retrieved:
                if chunk.get("source_id") not in existing_ids:
                    all_sources.append(chunk)
                    existing_ids.add(chunk.get("source_id"))

            if status_callback:
                status_callback(
                    f"📄 Trovati {len(retrieved)} chunks. Riprendo generazione..."
                )

            # Riprendi la generazione con il contesto iniettato
            past_queries = [e.cerca_query for e in result.cerca_events] + [cerca_query]
            gen_result = self.resume_after_cerca(
                question,
                accumulated_text,
                all_sources,
                past_queries=past_queries,
                stream_callback=stream_callback,
            )

            # Il nuovo testo viene concatenato
            if gen_result["text"]:
                accumulated_text = gen_result["text"]

            event.resumed = True
            result.cerca_events.append(event)

        # ── Fallback: se il modello non ha emesso <CERCA:> ─────────
        # llama3-8B potrebbe rispondere direttamente senza usare il tag.
        # In questo caso, facciamo una retrieval diretta con la domanda
        # originale per garantire che l'audit abbia evidenze.
        if iteration == 0 and not all_sources:
            if status_callback:
                status_callback(
                    "⚠️ No CERCA detected — Fallback: searching with original question..."
                )

            retrieved = self.retriever.query(question, top_k=self.top_k)
            all_sources = retrieved

            # Registra come evento CERCA "fallback"
            fallback_event = CercaEvent(
                iteration=1,
                cerca_query=f"[FALLBACK] {question}",
                partial_text=accumulated_text,
                retrieved_chunks=retrieved,
                resumed=True,
            )
            result.cerca_events.append(fallback_event)
            iteration = 1

            if status_callback:
                status_callback(
                    f"📄 Fallback: trovati {len(retrieved)} chunks. "
                    f"Rigenerando con contesto..."
                )

            # Rigenera la risposta con il contesto recuperato
            gen_result = self.resume_after_cerca(
                question, "", all_sources, stream_callback=stream_callback
            )
            accumulated_text = gen_result["text"]

        result.intermediate_text = accumulated_text
        result.all_sources = all_sources
        result.total_iterations = iteration

        # ── Step 3: Raffinamento ──────────────────────────────────
        if status_callback:
            status_callback("✨ Raffinamento del testo intermedio...")

        result.final_response = self._refine_response(accumulated_text)

        if status_callback:
            status_callback(
                f"✅ Completato con {iteration} ricerche CERCA"
            )

        return result

    # ────────────────────────────────────────────────────────────────
    # Raffinamento
    # ────────────────────────────────────────────────────────────────

    def _refine_response(self, intermediate_text: str) -> str:
        """
        Usa un secondo prompt per riscrivere il testo in forma pulita.

        Preserva citazioni e fatti, rimuove artefatti e ripetizioni.
        """
        # Costruisci le istruzioni base
        system_rules = settings.REFINEMENT_PROMPT.format(
            intermediate_text=intermediate_text,
        )

        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_rules}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Please finalize the draft as instructed.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        try:
            refined = self.generator.generate(prompt)
            # Se il raffinamento fallisce o è vuoto, usa il testo originale
            if refined.strip():
                return refined.strip()
        except Exception:
            pass

        return intermediate_text
