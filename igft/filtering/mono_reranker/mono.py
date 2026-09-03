"""MonoT5 post-retrieval reranking.

The ``MonoT5`` class wraps ``castorini/monot5-base-msmarco`` while
``rerank_ranking_file`` consumes the TSV output of the retrieval stage and
rewrites it with MonoT5 re-ranked document ids.
"""

from __future__ import annotations

import csv
import json
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from igft.filtering.common import load_corpus_jsonl


class MonoT5:
    """MonoT5 relevance model."""

    def __init__(self, model_name: str = "castorini/monot5-base-msmarco") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def score(self, query: str, document: str) -> float:
        input_text = f"Query: {query} Document: {document}"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, return_dict_in_generate=True, output_scores=True, output_logits=True
            )

        decoded = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        if decoded == "false":
            return 0.0

        logits = outputs.logits[0].cpu().numpy().flatten()
        true_logit = logits[1176]
        false_logit = logits[6136]
        return float(np.exp(true_logit) / (np.exp(false_logit) + np.exp(true_logit)))


def read_ranking_tsv(path: str) -> List[List[str]]:
    """Read a retrieval ranking file as ``[query_id, corpus_line_id]`` pairs."""
    pairs = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, desc=f"Loading {path}"):
            if len(row) < 2 or row[2] == "score":
                continue
            pairs.append([row[0], row[1]])
    return pairs


def line_id_to_corpus_id(corpus_file: str) -> Dict[str, str]:
    """Map each corpus file line number to its BEIR ``_id``."""
    mapping = {}
    with open(corpus_file, "r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(tqdm(f, desc="Indexing corpus lines")):
            item = json.loads(line)
            mapping[str(idx)] = item["_id"]
    return mapping


def rerank_ranking_file(
    corpus_file: str, query_file: str, ranking_file: str, reranker: MonoT5
) -> None:
    """Rerank every candidate list in ``ranking_file`` and overwrite it in place."""
    corpus = load_corpus_jsonl(corpus_file)
    queries = load_corpus_jsonl(query_file)  # queries.jsonl shares the _id/text layout
    rankings = read_ranking_tsv(ranking_file)
    line2corpus = line_id_to_corpus_id(corpus_file)

    grouped: Dict[str, List[str]] = {}
    for qid, line_id in rankings:
        grouped.setdefault(qid, []).append(line_id)

    reranked_rows: List[List[str]] = []
    for qid, line_ids in tqdm(grouped.items(), desc="Reranking"):
        scored = []
        for line_id in line_ids:
            doc = corpus[line2corpus[line_id]]
            score = reranker.score(queries[qid], doc)
            scored.append((line_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        for rank, (line_id, _) in enumerate(scored, start=1):
            reranked_rows.append([qid, line_id, str(rank)])

    with open(ranking_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(reranked_rows)
    print(f"Reranked results saved to {ranking_file}.")
