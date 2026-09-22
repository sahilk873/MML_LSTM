import argparse
import ast
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd

from polypharmacy import data as data_lib
from polypharmacy import llm_classifier
from polypharmacy import mechanism
from polypharmacy import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify (drug set, disease) rows into mechanism categories using OpenAI Responses API."
    )
    parser.add_argument("--indications-csv", default="indications_norm_dedup.csv")
    parser.add_argument("--contraindications-csv", default="contraindications_norm_dedup.csv")
    parser.add_argument("--output-dir", default="artifacts/mechanism_labels")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel OpenAI requests for row classification.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached classifications if present (default: true).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _parse_optional_list_column(df: pd.DataFrame, column: str) -> List[List[str]]:
    if column not in df.columns:
        return [[] for _ in range(len(df))]
    return df[column].apply(data_lib.parse_list_column).tolist()


def _safe_condition_label(row: pd.Series) -> str:
    for candidate in ("condition_id_norm_norm_label", "condition_id_norm_label"):
        if candidate in row and isinstance(row[candidate], str) and row[candidate].strip():
            return row[candidate].strip()
    return ""


def _load_source_df(path: str, source_file: str, label: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    primary = df["primary_drug_id_norm"].apply(data_lib.parse_list_column)
    secondary = df["secondary_drug_id_norm"].apply(data_lib.parse_list_column)
    primary_labels = _parse_optional_list_column(df, "primary_drug_id_norm_label")
    secondary_labels = _parse_optional_list_column(df, "secondary_drug_id_norm_label")

    rows = []
    for idx, row in df.iterrows():
        drug_ids = sorted(data_lib.normalize_id_list(primary.iloc[idx] + secondary.iloc[idx]))
        if not drug_ids:
            continue
        condition_id = str(row["condition_id_norm"]).strip()
        if not condition_id or condition_id.lower() in {"nan", "none"}:
            continue
        if condition_id.startswith("Error"):
            continue

        drug_labels = data_lib.normalize_id_list(primary_labels[idx] + secondary_labels[idx])
        example_key = mechanism.canonical_example_key(drug_ids, condition_id)
        rows.append(
            {
                "example_key": example_key,
                "drug_set": drug_ids,
                "condition_id_norm": condition_id,
                "source_label_binary": int(label),
                "source_file": source_file,
                "drug_labels": drug_labels,
                "condition_label": _safe_condition_label(row),
            }
        )

    return pd.DataFrame(rows)


def load_combined_dataset(indications_csv: str, contraindications_csv: str) -> pd.DataFrame:
    indications = _load_source_df(indications_csv, source_file="indications", label=1)
    contraindications = _load_source_df(
        contraindications_csv, source_file="contraindications", label=0
    )
    combined = pd.concat([indications, contraindications], ignore_index=True)
    return combined


def _stringify_list_column(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].apply(lambda values: str(list(values)))


def _compute_summary(
    labeled_df: pd.DataFrame, failures_df: pd.DataFrame, prompt_version: str, model: str
) -> Dict[str, object]:
    counts = labeled_df["mechanism_category"].value_counts().to_dict()
    confidences = labeled_df["mechanism_confidence"].astype(float)
    summary = {
        "model": model,
        "prompt_version": prompt_version,
        "num_rows": int(len(labeled_df)),
        "num_failures": int(len(failures_df)),
        "category_counts": {k: int(v) for k, v in counts.items()},
        "confidence": {
            "mean": float(confidences.mean()) if len(confidences) else float("nan"),
            "median": float(confidences.median()) if len(confidences) else float("nan"),
            "min": float(confidences.min()) if len(confidences) else float("nan"),
            "max": float(confidences.max()) if len(confidences) else float("nan"),
        },
        "needs_human_review_rate": float(
            labeled_df["needs_human_review"].astype(bool).mean() if len(labeled_df) else 0.0
        ),
    }
    return summary


def main() -> None:
    args = parse_args()
    utils.set_seeds(args.seed)
    utils.ensure_dir(args.output_dir)

    llm_classifier.require_openai_api_key()

    combined_df = load_combined_dataset(args.indications_csv, args.contraindications_csv)
    if args.max_rows is not None and len(combined_df) > args.max_rows:
        combined_df = combined_df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)

    combined_path = os.path.join(args.output_dir, "combined_source_dataset.csv")
    to_save = combined_df.copy()
    to_save["drug_set"] = _stringify_list_column(to_save, "drug_set")
    to_save["drug_labels"] = _stringify_list_column(to_save, "drug_labels")
    to_save.to_csv(combined_path, index=False)

    unique_df = combined_df.drop_duplicates(subset=["example_key"]).reset_index(drop=True)
    cache_path = os.path.join(args.output_dir, "classification_cache.jsonl")
    classifier = llm_classifier.OpenAIMechanismClassifier(
        model=args.model,
        prompt_version=args.prompt_version,
        cache_path=cache_path if args.resume else None,
    )

    annotations: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    rows = unique_df.to_dict(orient="records")
    force_flag = args.force or not args.resume

    def _run_one(row: Dict[str, object]) -> Dict[str, object]:
        result = classifier.classify_row(row, force=force_flag)
        cls = result["classification"]
        return {
            "example_key": row["example_key"],
            "drug_set": row["drug_set"],
            "condition_id_norm": row["condition_id_norm"],
            "mechanism_category": cls.category,
            "mechanism_confidence": float(cls.confidence),
            "needs_human_review": bool(cls.needs_human_review),
            "rationale_short": cls.rationale_short,
            "llm_model": args.model,
            "prompt_version": args.prompt_version,
            "classified_at_utc": mechanism.utc_now_iso(),
            "from_cache": bool(result["cached"]),
        }

    print(f"Classifying {len(rows)} unique examples with workers={max(1, args.workers)}")
    if max(1, args.workers) == 1:
        for idx, row in enumerate(rows, start=1):
            try:
                annotations.append(_run_one(row))
            except Exception as exc:
                failures.append(
                    {
                        "example_key": row["example_key"],
                        "condition_id_norm": row["condition_id_norm"],
                        "drug_set": row["drug_set"],
                        "error": str(exc),
                    }
                )
            if idx % 100 == 0 or idx == len(rows):
                print(f"Progress: {idx}/{len(rows)}")
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            future_to_row = {pool.submit(_run_one, row): row for row in rows}
            completed = 0
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                completed += 1
                try:
                    annotations.append(future.result())
                except Exception as exc:
                    failures.append(
                        {
                            "example_key": row["example_key"],
                            "condition_id_norm": row["condition_id_norm"],
                            "drug_set": row["drug_set"],
                            "error": str(exc),
                        }
                    )
                if completed % 100 == 0 or completed == len(rows):
                    print(f"Progress: {completed}/{len(rows)}")

    annotations_df = pd.DataFrame(annotations)
    if len(annotations_df) == 0:
        raise RuntimeError("No rows were successfully classified")

    labeled_df = combined_df.merge(
        annotations_df[
            [
                "example_key",
                "mechanism_category",
                "mechanism_confidence",
                "needs_human_review",
                "rationale_short",
                "llm_model",
                "prompt_version",
                "classified_at_utc",
            ]
        ],
        on="example_key",
        how="left",
    )

    annotations_path = os.path.join(args.output_dir, "mechanism_annotations.csv")
    labeled_path = os.path.join(args.output_dir, "mechanism_labeled_dataset.csv")
    failures_path = os.path.join(args.output_dir, "classification_failures.csv")
    summary_path = os.path.join(args.output_dir, "classification_summary.json")

    annotations_out = annotations_df.copy()
    annotations_out["drug_set"] = _stringify_list_column(annotations_out, "drug_set")
    annotations_out.to_csv(annotations_path, index=False)

    labeled_out = labeled_df.copy()
    labeled_out["drug_set"] = _stringify_list_column(labeled_out, "drug_set")
    labeled_out["drug_labels"] = _stringify_list_column(labeled_out, "drug_labels")
    labeled_out.to_csv(labeled_path, index=False)

    failures_df = pd.DataFrame(failures)
    if len(failures_df):
        failures_df["drug_set"] = failures_df["drug_set"].apply(lambda values: str(list(values)))
    failures_df.to_csv(failures_path, index=False)

    summary = _compute_summary(labeled_df.dropna(subset=["mechanism_category"]), failures_df, args.prompt_version, args.model)
    utils.save_json(summary_path, summary)

    print(f"Wrote: {annotations_path}")
    print(f"Wrote: {labeled_path}")
    print(f"Wrote: {failures_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
