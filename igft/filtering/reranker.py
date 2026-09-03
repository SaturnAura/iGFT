"""Post-retrieval reranking entry point (MonoT5).

Example:

    python -m igft.filtering.reranker \\
        --corpus third_party/SPTAR/dense_retrieval/datasets/raw/beir/fiqa/corpus.jsonl \\
        --query third_party/SPTAR/dense_retrieval/datasets/raw/beir/fiqa/queries.jsonl \\
        --ranking outputs/colbert_ranking.tsv
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank a retrieval ranking file with MonoT5.")
    parser.add_argument("--corpus", required=True, help="BEIR corpus JSONL file.")
    parser.add_argument("--query", required=True, help="BEIR queries JSONL file.")
    parser.add_argument("--ranking", required=True, help="Retrieval ranking TSV (rewritten in place).")
    parser.add_argument(
        "--reranker-path",
        default="castorini/monot5-base-msmarco",
        help="HuggingFace model id or local MonoT5 checkpoint.",
    )
    args = parser.parse_args()

    from igft.filtering.mono_reranker.mono import MonoT5, rerank_ranking_file

    reranker = MonoT5(args.reranker_path)
    rerank_ranking_file(args.corpus, args.query, args.ranking, reranker)


if __name__ == "__main__":
    main()
