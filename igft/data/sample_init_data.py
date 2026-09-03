"""Sample an initial low-resource training set from a BEIR dataset.

The sampled ``(document -> gold query)`` pairs are written in the JSON
format expected by LLaMA-Factory's ``dataset_info.json`` so that they can
be used directly for supervised fine-tuning of the query generator.

Examples:

    # sample 50 pairs from FiQA for the low-resource setting
    python -m igft.data.sample_init_data --dataset fiqa --num 50

    # use the whole training split (fully-supervised setting)
    python -m igft.data.sample_init_data --dataset fiqa --num -1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

from igft.data.io import read_jsonl, read_tsv_pairs, write_json, write_tsv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEIR_ROOT = (
    REPO_ROOT / "third_party" / "SPTAR" / "dense_retrieval" / "datasets" / "raw" / "beir"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "third_party" / "LLaMA-Factory" / "data"
DEFAULT_SPTAR_ROOT = REPO_ROOT / "third_party" / "SPTAR"


def build_llm_training_data(
    corpus_file: Path,
    query_file: Path,
    qrels_file: Path,
    sample_num: int,
    shuffle: bool = False,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Build ``{document: instruction, gold query: output}`` training pairs.

    The fields follow the LLaMA-Factory convention:
    ``prompt=instruction``, ``query=input``, ``response=output``.
    """
    corpus = read_jsonl_as_text(corpus_file)
    queries = read_jsonl_as_text(query_file)
    query_doc_pairs = read_tsv_pairs(qrels_file)

    if sample_num == -1:
        sample_num = len(query_doc_pairs)
        print(f"Sampling all {sample_num} training pairs from {corpus_file.parent.name}.")
    elif shuffle:
        rng = random.Random(seed)
        query_doc_pairs = rng.sample(query_doc_pairs, sample_num)
        print(f"Randomly sampled {sample_num} training pairs from {corpus_file.parent.name}.")
    else:
        print(f"Using the first {sample_num} training pairs from {corpus_file.parent.name}.")

    records = []
    sampled_pairs = query_doc_pairs[:sample_num]
    for qid, cid in sampled_pairs:
        records.append(
            {
                "query_id": qid,
                "corpus_id": cid,
                "instruction": corpus[cid],
                "input": "",
                "output": queries[qid],
            }
        )
    return records


def build_prompt_tuning_qrels(records: List[Dict[str, str]]) -> List[List[str]]:
    """Convert the sampled gold pairs into the qrels TSV consumed by SPTAR."""
    return [[record["query_id"], record["corpus_id"], "1"] for record in records]


def read_jsonl_as_text(path: Path) -> Dict[str, str]:
    """Read a BEIR ``*.jsonl`` file into ``{_id: text}``."""
    mapping: Dict[str, str] = {}
    for item in read_jsonl(path):
        mapping[item["_id"]] = item["text"]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample initial (document, query) pairs for iGFT query-generator SFT."
    )
    parser.add_argument("--dataset", required=True, help="BEIR dataset name, e.g. fiqa / msmarco / nq.")
    parser.add_argument(
        "--num",
        type=int,
        required=True,
        help="Number of training pairs to sample. Use -1 to sample the whole training split.",
    )
    parser.add_argument(
        "--beir-root",
        type=Path,
        default=DEFAULT_BEIR_ROOT,
        help="Directory that contains the downloaded BEIR dataset folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. By default writes to LLaMA-Factory's data directory.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly sample the training pairs instead of taking the first N rows.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used with --shuffle.")
    parser.add_argument(
        "--sptar-root",
        type=Path,
        default=DEFAULT_SPTAR_ROOT,
        help="SPTAR checkout root used for the pseudo-query data directory.",
    )
    args = parser.parse_args()

    dataset_dir = args.beir_root / args.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Please download the dataset first (see scripts/download_beir.sh)."
        )

    records = build_llm_training_data(
        corpus_file=dataset_dir / "corpus.jsonl",
        query_file=dataset_dir / "queries.jsonl",
        qrels_file=dataset_dir / "qrels" / "train.tsv",
        sample_num=args.num,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    output_path = args.output or DEFAULT_OUTPUT_DIR / f"{args.dataset}_train.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(records, output_path)

    # SPTAR's retrieval stage needs the same gold pairs as a BEIR-style qrels TSV.
    if args.num != -1:
        qrels_path = (
            args.sptar_root
            / "pseudo_query"
            / "data"
            / f"{args.dataset}_{args.num}"
            / f"prompt_tuning_{args.num}.tsv"
        )
        qrels_path.parent.mkdir(parents=True, exist_ok=True)
        write_tsv(
            build_prompt_tuning_qrels(records),
            qrels_path,
            header=["query-id", "corpus-id", "score"],
        )


if __name__ == "__main__":
    main()
