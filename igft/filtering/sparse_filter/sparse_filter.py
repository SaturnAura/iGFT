"""Sparse (BM25) quality filter for pseudo queries.

For every pseudo query, the expected target document is mixed into a
random candidate pool and BM25 is used to retrieve against it. The query
keeps its BM25 score only when the target document is ranked first;
otherwise the pair is treated as a filtering failure (score 0).
"""

from __future__ import annotations

from typing import Dict, List

from rank_bm25 import BM25Okapi

from igft.filtering.common import load_corpus_jsonl, load_pseudo_queries, sample_documents, write_json


def bm25_score(query: str, gold_document: str, candidate_documents: List[str]) -> float:
    """Return the BM25 score of the gold document iff it is ranked first."""
    documents = [gold_document] + candidate_documents
    tokenized = [doc.split() for doc in documents]
    scores = list(BM25Okapi(tokenized).get_scores(query.split()))
    return scores[0] if scores.index(max(scores)) == 0 else 0.0


def score_pseudo_queries(
    pseudo_query_file: str,
    corpus_file: str,
    output_file: str,
    candidate_num: int = 500,
) -> None:
    """Score every pseudo query with BM25 and write the results to JSON."""
    corpus: Dict[str, str] = load_corpus_jsonl(corpus_file)
    pseudo_queries = load_pseudo_queries(pseudo_query_file)
    candidates = sample_documents(corpus, candidate_num)

    results = []
    for item in pseudo_queries:
        score = bm25_score(item["pseudo_query"], item["corpus"], candidates)
        results.append(
            {
                "score": score,
                "pseudo_query": item["pseudo_query"],
                "corpus": item["corpus"],
                "cid": item["cid"],
            }
        )
    write_json(results, output_file)
