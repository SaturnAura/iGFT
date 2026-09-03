"""Convert filtered pseudo queries into ColBERT / SPTAR training files.

The original BEIR-style ``(pseudo query, target doc id)`` pairs produced
by the filtering stage are converted into the two files expected by the
SPTAR data loader:

* ``weak_queries_<train_num>_<exp_name>.jsonl``  -- pseudo queries
* ``weak_train_<train_num>_<exp_name>.tsv``      -- query-id / corpus-id

Files are written under ``pseudo_query/data/<dataset>_<train_num>/<weak_num>/``
inside the vendored SPTAR checkout.

Example:

    python -m igft.data.convert_colbert_data \\
        --dataset fiqa \\
        --pseudo outputs/fiqa_filtered.json \\
        --exp-name my_exp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from igft.data.io import read_json, write_jsonl, write_tsv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "third_party" / "SPTAR" / "pseudo_query" / "data"
WEAK_QUERY_ID_OFFSET = 4000001


def sort_by_score(pseudo_queries: List[dict]) -> List[dict]:
    """Sort pseudo queries by their filter score (highest first) if present."""
    if pseudo_queries and "score" in pseudo_queries[0]:
        return sorted(pseudo_queries, key=lambda x: x["score"], reverse=True)
    return pseudo_queries


def build_colbert_files(
    pseudo_queries: List[dict],
) -> Tuple[List[List[str]], List[dict]]:
    """Map each filtered pseudo query to a fresh query id + its target corpus id."""
    tsv_rows: List[List[str]] = []
    jsonl_rows: List[dict] = []

    next_query_id = WEAK_QUERY_ID_OFFSET
    for item in pseudo_queries:
        query_id = next_query_id
        next_query_id += 1
        tsv_rows.append([str(query_id), str(item["cid"]), "1"])
        jsonl_rows.append({"_id": query_id, "text": item["pseudo_query"], "metadata": {}})
    return tsv_rows, jsonl_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert filtered pseudo queries into SPTAR/ColBERT training data."
    )
    parser.add_argument("--dataset", required=True, help="BEIR dataset name, e.g. fiqa / msmarco.")
    parser.add_argument(
        "--pseudo",
        required=True,
        help="Filtered pseudo-query JSON file (list of dicts with pseudo_query/cid/score).",
    )
    parser.add_argument(
        "--exp-name",
        required=True,
        help="Experiment name; it becomes part of the output file names and must match "
        "the --exp-name passed to SPTAR's gen_data_for_colbert.py.",
    )
    parser.add_argument("--train-num", type=int, default=50, help="Number of gold training queries (default: 50).")
    parser.add_argument("--weak-num", type=str, default="100k", help="Weak query pool size tag (default: 100k).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root of the SPTAR pseudo-query data directory.",
    )
    args = parser.parse_args()

    pseudo_queries = sort_by_score(read_json(args.pseudo))
    if not pseudo_queries:
        raise ValueError("The pseudo-query file is empty or does not contain a JSON list.")

    output_dir = args.output_root / f"{args.dataset}_{args.train_num}" / args.weak_num
    output_dir.mkdir(parents=True, exist_ok=True)

    tsv_rows, jsonl_rows = build_colbert_files(pseudo_queries)
    write_tsv(
        tsv_rows,
        output_dir / f"weak_train_{args.train_num}_{args.exp_name}.tsv",
        header=["query-id", "corpus-id", "score"],
    )
    write_jsonl(jsonl_rows, output_dir / f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")


if __name__ == "__main__":
    main()
