"""Entry point of the multi-stage pseudo-query filtering modules.

Example:

    python -m igft.filtering.filter \\
        --pseudo outputs/pseudo_queries.json \\
        --corpus third_party/SPTAR/dense_retrieval/datasets/raw/beir/fiqa/corpus.jsonl \\
        --mode BM25 --candidate-num 500 --output outputs/fiqa_bm25.json
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Score pseudo queries with sparse/dense quality filters.")
    parser.add_argument("--pseudo", required=True, help="Generated pseudo-query JSON file.")
    parser.add_argument("--corpus", required=True, help="BEIR corpus JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSON file with per-query scores.")
    parser.add_argument(
        "--mode",
        choices=["BM25", "DPR", "ColBERT", "MonoT5"],
        default="BM25",
        help="Filtering backbone (default: BM25).",
    )
    parser.add_argument(
        "--candidate-num",
        type=int,
        default=500,
        help="Number of random distractor documents mixed into the candidate set (default: 500).",
    )
    args = parser.parse_args()

    if args.mode == "BM25":
        from igft.filtering.sparse_filter.sparse_filter import score_pseudo_queries as sparse_score

        sparse_score(args.pseudo, args.corpus, args.output, candidate_num=args.candidate_num)
    else:
        from igft.filtering.dense_filter.dense_filter import score_pseudo_queries as dense_score

        dense_score(args.pseudo, args.corpus, args.output, mode=args.mode, candidate_num=args.candidate_num)


if __name__ == "__main__":
    main()
