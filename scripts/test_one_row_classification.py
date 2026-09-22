#!/usr/bin/env python3
"""Smoke test: classify a single dataset row with OpenAI mechanism classifier."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polypharmacy import data as data_lib
from polypharmacy import llm_classifier
from polypharmacy import mechanism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify one row from indications/contraindications CSV via OpenAI."
    )
    parser.add_argument("--indications-csv", default="indications_norm_dedup.csv")
    parser.add_argument("--contraindications-csv", default="contraindications_norm_dedup.csv")
    parser.add_argument(
        "--source",
        choices=["indications", "contraindications"],
        default="indications",
        help="Which CSV to sample from.",
    )
    parser.add_argument("--row-index", type=int, default=0, help="0-based row index in chosen source CSV.")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--force", action="store_true", help="Ignore cache and force API call.")
    return parser.parse_args()


def _safe_condition_label(row: pd.Series) -> str:
    for candidate in ("condition_id_norm_norm_label", "condition_id_norm_label"):
        if candidate in row and isinstance(row[candidate], str) and row[candidate].strip():
            return row[candidate].strip()
    return ""


def _safe_labels(row: pd.Series, column: str) -> list[str]:
    if column not in row.index:
        return []
    return data_lib.parse_list_column(row[column])


def build_row_payload(df_row: pd.Series, source: str, label: int) -> dict:
    primary = data_lib.parse_list_column(df_row["primary_drug_id_norm"])
    secondary = data_lib.parse_list_column(df_row["secondary_drug_id_norm"])
    drug_set = sorted(data_lib.normalize_id_list(primary + secondary))

    condition_id = str(df_row["condition_id_norm"]).strip()
    if not drug_set:
        raise ValueError("Selected row has empty parsed drug_set")
    if not condition_id or condition_id.lower() in {"nan", "none"}:
        raise ValueError("Selected row has invalid condition_id_norm")

    primary_labels = _safe_labels(df_row, "primary_drug_id_norm_label")
    secondary_labels = _safe_labels(df_row, "secondary_drug_id_norm_label")

    return {
        "example_key": mechanism.canonical_example_key(drug_set, condition_id),
        "drug_set": drug_set,
        "condition_id_norm": condition_id,
        "source_label_binary": int(label),
        "source_file": source,
        "drug_labels": data_lib.normalize_id_list(primary_labels + secondary_labels),
        "condition_label": _safe_condition_label(df_row),
    }


def main() -> None:
    args = parse_args()
    llm_classifier.require_openai_api_key()

    source_to_path = {
        "indications": args.indications_csv,
        "contraindications": args.contraindications_csv,
    }
    source_to_label = {"indications": 1, "contraindications": 0}

    source_path = source_to_path[args.source]
    source_label = source_to_label[args.source]

    df = pd.read_csv(source_path)
    if args.row_index < 0 or args.row_index >= len(df):
        raise IndexError(f"--row-index must be in [0, {len(df)-1}] for {source_path}")

    row_payload = build_row_payload(df.iloc[args.row_index], source=args.source, label=source_label)

    classifier = llm_classifier.OpenAIMechanismClassifier(
        model=args.model,
        prompt_version=args.prompt_version,
        cache_path=args.cache_path,
    )
    result = classifier.classify_row(row_payload, force=args.force)

    output = {
        "source_csv": source_path,
        "source": args.source,
        "row_index": args.row_index,
        "example_key": result["example_key"],
        "cached": bool(result["cached"]),
        "classification": {
            "category": result["classification"].category,
            "confidence": result["classification"].confidence,
            "needs_human_review": result["classification"].needs_human_review,
            "rationale_short": result["classification"].rationale_short,
        },
        "input": {
            "drug_set": row_payload["drug_set"],
            "condition_id_norm": row_payload["condition_id_norm"],
            "source_label_binary": row_payload["source_label_binary"],
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
