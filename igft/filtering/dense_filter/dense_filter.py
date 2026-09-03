"""Dense quality filters for pseudo queries.

This module evaluates a generated query against its expected target
document using one of several neural scoring backbones:

* ``DPR`` -- a Sentence-BERT based bi-encoder;
* ``ColBERT`` -- a BERT mean-pooling encoder;
* ``MonoT5`` -- a T5 sequence-to-sequence relevance scorer.

For bi-encoder backbones, the target document is ranked inside a random
candidate pool and the pseudo query is kept only if the target wins.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

from igft.filtering.common import load_corpus_jsonl, load_pseudo_queries, sample_documents, write_json


class DPRScorer:
    """DPR-style bi-encoder that ranks the gold document among candidates."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        model = DRES(SentenceBERT(model_name), batch_size=256, corpus_chunk_size=100000)
        self.retriever = EvaluateRetrieval(model, k_values=[10], score_function="cos_sim")

    def score(self, query: str, gold_document: str, candidate_documents: List[str]) -> float:
        corpus = {str(i): {"text": doc} for i, doc in enumerate([gold_document] + candidate_documents)}
        queries = {"qid": query, "dummy": "filler"}
        result = self.retriever.retrieve(corpus, queries)
        scores = list(result["qid"].values())
        return scores[0] if scores.index(max(scores)) == 0 else 0.0


class BertDenseScorer:
    """BERT mean-pooling encoder used for the ColBERT filtering mode."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def _embed(self, sentences: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            token_embeddings = self.model(**encoded)[0]
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def score(self, query: str, gold_document: str, candidate_documents: List[str]) -> float:
        query_emb = self._embed([query])
        doc_emb = self._embed([gold_document] + candidate_documents)
        similarities = F.cosine_similarity(query_emb.expand_as(doc_emb), doc_emb, dim=1).tolist()
        return similarities[0] if similarities.index(max(similarities)) == 0 else 0.0


class MonoT5Scorer:
    """MonoT5 relevance scorer (``castorini/monot5-base-msmarco``)."""

    def __init__(self, model_name: str = "castorini/monot5-base-msmarco") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def score(self, query: str, gold_document: str, candidate_documents: List[str]) -> float:
        input_text = f"Query: {query} Document: {gold_document}"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, return_dict_in_generate=True, output_scores=True, output_logits=True
            )

        decoded = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        if decoded == "false":
            return 0.0

        logits = outputs.logits[0].cpu().numpy().flatten()
        true_logit = logits[1176]  # token id of "true"
        false_logit = logits[6136]  # token id of "false"
        return float(np.exp(true_logit) / (np.exp(false_logit) + np.exp(true_logit)))


def _build_scorer(mode: str):
    if mode == "DPR":
        return DPRScorer()
    if mode == "ColBERT":
        return BertDenseScorer()
    if mode == "MonoT5":
        return MonoT5Scorer()
    raise ValueError(f"Unknown dense filter mode: {mode}")


def score_pseudo_queries(
    pseudo_query_file: str,
    corpus_file: str,
    output_file: str,
    mode: str = "DPR",
    candidate_num: int = 500,
) -> None:
    """Score pseudo queries with the requested dense backbone and write results."""
    corpus: Dict[str, str] = load_corpus_jsonl(corpus_file)
    pseudo_queries = load_pseudo_queries(pseudo_query_file)
    candidates = sample_documents(corpus, candidate_num)
    scorer = _build_scorer(mode)

    results = []
    for item in pseudo_queries:
        generated_query = item["pseudo_query"]
        score = scorer.score(generated_query, item["corpus"], candidates)
        results.append(
            {
                "score": score,
                "pseudo_query": generated_query,
                "corpus": item["corpus"],
                "cid": item["cid"],
            }
        )
    write_json(results, output_file)
