"""Shared file I/O and data structures for pseudo-query filtering.

Pseudo-query files produced by the query generator are JSON lists, each item
containing at least the keys ``pseudo_query``, ``cid`` and ``corpus``. BEIR
corpus files are JSONL with ``_id`` / ``title`` / ``text`` fields.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


def load_pseudo_queries(path: str | Path) -> List[dict]:
    """Load the JSON list emitted by the pseudo-query generation stage."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_corpus_jsonl(path: str | Path) -> Dict[str, str]:
    """Load a BEIR corpus JSONL file into ``{_id: text}``."""
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in tqdm(f, desc=f"Indexing {Path(path).name}"):
            item = json.loads(line)
            mapping[item["_id"]] = item["text"]
    return mapping


def sample_documents(corpus: Dict[str, str], num: int, seed: int = 42) -> List[str]:
    """Randomly sample ``num`` document texts to serve as candidate distractors."""
    if num == 0:
        return []
    if num > len(corpus):
        raise ValueError(
            f"candidate_num={num} is larger than the corpus size ({len(corpus)}). "
            "Decrease --candidate-num."
        )
    rng = random.Random(seed)
    sampled_ids = rng.sample(list(corpus), num)
    return [corpus[cid] for cid in sampled_ids]


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write the scored pseudo queries as an indented JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    print(f"Results saved to {path}.")
