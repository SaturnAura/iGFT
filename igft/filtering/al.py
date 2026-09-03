"""Active-learning based pseudo-query filtering entry point.

Train the loss predictor first, then reuse it to score every pseudo query:

    # 1) train
    python -m igft.filtering.al \\
        --mode train \\
        --pseudo outputs/pseudo_queries.json \\
        --corpus third_party/SPTAR/dense_retrieval/datasets/raw/beir/fiqa/corpus.jsonl

    # 2) predict and filter
    python -m igft.filtering.al \\
        --pseudo outputs/pseudo_queries.json \\
        --corpus third_party/SPTAR/dense_retrieval/datasets/raw/beir/fiqa/corpus.jsonl \\
        --output outputs/fiqa_al_scores.json
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Active-learning loss predictor for pseudo queries.")
    parser.add_argument("--pseudo", required=True, help="Generated pseudo-query JSON file.")
    parser.add_argument("--corpus", required=True, help="BEIR corpus JSONL file.")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="predict",
        help="Train the loss predictor or run prediction (default: predict).",
    )
    parser.add_argument("--output", help="JSON file to write predicted scores (required for predict).")
    parser.add_argument(
        "--lossnet-path",
        default="lossnet",
        help="Checkpoint prefix, saved as <path>.pt (default: lossnet).",
    )
    parser.add_argument(
        "--base-retriever",
        default="bert-base-uncased",
        help="HuggingFace model used as the base retriever encoder.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10).")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size (default: 4).")
    args = parser.parse_args()

    from igft.filtering.active_learning.lossnet import test_main, train_main

    if args.mode == "train":
        train_main(
            args.pseudo,
            args.corpus,
            lossnet_path=args.lossnet_path,
            base_retriever_path=args.base_retriever,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        if not args.output:
            parser.error("--output is required when --mode predict")
        test_main(
            args.pseudo,
            args.corpus,
            args.output,
            lossnet_path=args.lossnet_path,
            base_retriever_path=args.base_retriever,
        )


if __name__ == "__main__":
    main()
