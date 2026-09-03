"""Shared file I/O helpers for iGFT data preparation.

All iGFT data scripts use the same lightweight helpers so that corpus /
query / pseudo-query formats are handled in exactly one place.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tqdm import tqdm


def read_json(path: str | Path) -> Any:
    """Read a whole JSON document (used for the pseudo-query JSON lists)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL file line by line into a list of dicts."""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in tqdm(f, desc=f"Loading {Path(path).name}"):
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def read_jsonl_as_dict(
    path: str | Path, key_field: str = "_id", text_field: str = "text"
) -> Dict[str, str]:
    """Read a BEIR-style JSONL file into ``{key: text}``."""
    mapping: Dict[str, str] = {}
    for item in tqdm(read_jsonl(path), desc=f"Indexing {Path(path).name}"):
        key = item[key_field]
        mapping[key] = item[text_field]
    return mapping


def read_tsv_pairs(path: str | Path) -> List[List[str]]:
    """Read a TSV file (skipping its header) into ``(id_a, id_b)`` pairs."""
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        for row in tqdm(reader, desc=f"Loading {Path(path).name}"):
            if len(row) >= 2:
                rows.append([row[0], row[1]])
    return rows


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write a Python object as an indented JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    print(f"Successfully saved to {path}!")


def write_jsonl(items: Iterable[Dict[str, Any]], path: str | Path) -> None:
    """Write a sequence of dicts as one JSON object per line."""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Successfully saved to {path}!")


def write_tsv(rows: Iterable[Iterable[Any]], path: str | Path, header: List[str] | None = None) -> None:
    """Write rows to a tab-separated file, optionally with a header."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if header:
            writer.writerow(header)
        writer.writerows(rows)
    print(f"Successfully saved to {path}!")
