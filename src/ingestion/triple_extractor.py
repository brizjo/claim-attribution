"""
Triple extractor — uses REBEL (Babelscape/rebel-large, English BART) by
default; auto-switches to mREBEL config when model name contains "mrebel".

A single regex-tokenised parser handles both output formats:
  REBEL  : <triplet> SUBJ <subj> OBJ <obj> RELATION ...
  mREBEL : <triplet> SUBJ <subj_type> OBJ <obj_type> RELATION ...
Any bracketed token (other than <triplet>/<relation>) is treated as a
segment separator, so the same state machine works for both.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Optional

from config import settings


class Triple(NamedTuple):
    subject: str
    predicate: str
    obj: str
    chunk_text: str
    source_file: str
    chunk_index: int


# Language tokens occasionally emitted by mBART-based mREBEL — skipped.
_LANG_TOKENS = {
    "tp_XX", "__en__", "__it__", "__es__", "__de__", "__fr__",
    "__pt__", "__ro__", "__nl__", "__pl__",
}


def _parse_rebel_output(text: str) -> list[tuple[str, str, str]]:
    """
    Parse REBEL / mREBEL seq2seq output into (subject, predicate, object).

    Models often emit special tokens without surrounding whitespace
    (e.g. ``tp_XX<triplet>`` glued).  Regex-tokenise so every bracketed
    marker becomes a standalone token.
    """
    triplets: list[tuple[str, str, str]] = []
    text = text.replace("<s>", " ").replace("<pad>", " ").replace("</s>", " ")
    for lang in _LANG_TOKENS:
        text = re.sub(rf"\b{re.escape(lang)}\b", " ", text)
    text = re.sub(r"(<[^>]+>)", r" \1 ", text)
    tokens = text.split()

    current = "x"
    subject = predicate = obj = ""

    for token in tokens:
        if token in ("<triplet>", "<relation>"):
            if predicate and subject and obj:
                triplets.append((subject.strip(), predicate.strip(), obj.strip()))
            current = "t"
            subject = predicate = obj = ""
        elif token.startswith("<") and token.endswith(">"):
            # Either <subj>/<obj> (REBEL) or <ENTITY_TYPE> (mREBEL typed).
            # Both act as segment separators driven by current state.
            if current in ("t", "o"):
                if predicate and subject and obj:
                    triplets.append((subject.strip(), predicate.strip(), obj.strip()))
                current = "s"
                obj = ""
            else:
                current = "o"
                predicate = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                obj += " " + token
            elif current == "o":
                predicate += " " + token

    if subject and predicate and obj:
        triplets.append((subject.strip(), predicate.strip(), obj.strip()))

    return triplets


class TripleExtractor:
    """Extracts triples using REBEL (BART) or mREBEL (mBART) — auto-detected."""

    def __init__(
        self,
        model_name: str = settings.REBEL_MODEL,
        src_lang: Optional[str] = settings.REBEL_SRC_LANG,
    ):
        self._model_name = model_name
        self._src_lang = src_lang
        self._is_multilingual = "mrebel" in model_name.lower()
        self._tokenizer = None
        self._model = None
        self._device = None
        self._decoder_start_id: Optional[int] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if self._is_multilingual:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                src_lang=self._src_lang or "en_XX",
                tgt_lang="tp_XX",
            )
            self._decoder_start_id = self._tokenizer.convert_tokens_to_ids("tp_XX")
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._decoder_start_id = None  # use model default

        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(self._device)
        self._model.eval()

    def extract(self, chunks: list[dict]) -> list[Triple]:
        """
        Run REBEL/mREBEL on the entire chunk list using minibatching of size
        REBEL_BATCH_SIZE.  num_return_sequences=3 keeps all 3 beam
        hypotheses, deduped per chunk on (subject, predicate, object).
        """
        import torch

        self._load()
        triples: list[Triple] = []
        texts = [c["text"] for c in chunks]

        batch_size = settings.REBEL_BATCH_SIZE
        max_new = settings.REBEL_MAX_LENGTH
        n_return = 3

        gen_kwargs = dict(
            max_new_tokens=max_new,
            num_beams=3,
            num_return_sequences=n_return,
            length_penalty=0.0,
            early_stopping=True,
            forced_bos_token_id=None,
        )
        if self._decoder_start_id is not None:
            gen_kwargs["decoder_start_token_id"] = self._decoder_start_id

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            batch_chunks = chunks[start:start + batch_size]

            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)

            with torch.no_grad():
                gen = self._model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )

            decoded_all = self._tokenizer.batch_decode(gen, skip_special_tokens=False)

            for ci, chunk in enumerate(batch_chunks):
                beams = decoded_all[ci * n_return:(ci + 1) * n_return]
                seen: set[tuple[str, str, str]] = set()
                for generated in beams:
                    for subj, pred, obj in _parse_rebel_output(generated):
                        key = (subj.lower(), pred.lower(), obj.lower())
                        if key in seen:
                            continue
                        seen.add(key)
                        triples.append(Triple(
                            subject=subj,
                            predicate=pred,
                            obj=obj,
                            chunk_text=chunk["text"],
                            source_file=chunk["source_file"],
                            chunk_index=chunk["chunk_index"],
                        ))

        return triples
