import argparse
import ast
import os
from typing import Dict, List

import pandas as pd

from polypharmacy import mechanism
from polypharmacy import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild refined ground truth using mechanism annotations."
    )
    parser.add_argument(
        "--labeled-dataset-csv",
        default="artifacts/mechanism_labels/mechanism_labeled_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/refined_gt",
    )
    parser.add_argument(
        "--keep-categories",
        default=mechanism.CATEGORY_MECHANISTICALLY_SYNERGISTIC,
        help="Comma-separated mechanism categories to keep.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--drop-needs-review", action="store_true", default=True)
    return parser.parse_args()


def _parse_drug_set(value: object) -> List[str]:
    if isinstance(value, list):
        return sorted(str(item) for item in value)
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return sorted(str(item) for item in parsed)
    raise ValueError(f"Unable to parse drug_set: {value!r}")


def _resolve_binary_label(df: pd.DataFrame) -> pd.Series:
    if "source_label_binary" in df.columns:
        return df["source_label_binary"].astype(int)
    if "label" in df.columns:
        return df["label"].astype(int)
    raise ValueError("Expected source_label_binary or label column")


def _dedupe_with_conflict_resolution(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["drug_set"] = df["drug_set"].apply(_parse_drug_set)
    df["drug_set_key"] = df["drug_set"].apply(tuple)
    deduped = (
        df.groupby(["drug_set_key", "condition_id_norm"], as_index=False)
        .agg(label=("label", "min"), drug_set=("drug_set", "first"))
        .drop(columns=["drug_set_key"])
    )
    return deduped


def _to_training_schema(df: pd.DataFrame) -> pd.DataFrame:
    def split_primary_secondary(drug_set: List[str]) -> tuple:
        if not drug_set:
            raise ValueError("Encountered empty drug_set while building training schema")
        primary = [drug_set[0]]
        secondary = drug_set[1:]
        return primary, secondary

    rows: List[Dict[str, object]] = []
    for row in df.itertuples(index=False):
        primary, secondary = split_primary_secondary(list(row.drug_set))
        rows.append(
            {
                "primary_drug_id_norm": str(primary),
                "secondary_drug_id_norm": str(secondary),
                "condition_id_norm": str(row.condition_id_norm),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    keep_categories = mechanism.categories_from_csv_arg(args.keep_categories)
    utils.ensure_dir(args.output_dir)

    labeled_df = pd.read_csv(args.labeled_dataset_csv)
    if "mechanism_category" not in labeled_df.columns:
        raise ValueError("Expected mechanism_category in labeled dataset")

    labeled_df = labeled_df.copy()
    labeled_df["label"] = _resolve_binary_label(labeled_df)
    labeled_df["mechanism_confidence"] = labeled_df["mechanism_confidence"].astype(float)
    labeled_df["needs_human_review"] = labeled_df["needs_human_review"].astype(bool)

    total_rows = len(labeled_df)
    kept = labeled_df[labeled_df["mechanism_category"].isin(keep_categories)].copy()
    dropped_by_category = int(total_rows - len(kept))

    kept = kept[kept["mechanism_confidence"] >= float(args.min_confidence)].copy()
    dropped_by_confidence = int(total_rows - dropped_by_category - len(kept))

    dropped_by_review = 0
    if args.drop_needs_review:
        before = len(kept)
        kept = kept[~kept["needs_human_review"]].copy()
        dropped_by_review = int(before - len(kept))

    refined = _dedupe_with_conflict_resolution(
        kept[["drug_set", "condition_id_norm", "label"]].copy()
    )

    refined_dataset_path = os.path.join(args.output_dir, "refined_dataset.csv")
    refined_indications_path = os.path.join(args.output_dir, "refined_indications.csv")
    refined_contraindications_path = os.path.join(args.output_dir, "refined_contraindications.csv")
    report_path = os.path.join(args.output_dir, "refinement_report.json")

    refined_out = refined.copy()
    refined_out["drug_set"] = refined_out["drug_set"].apply(lambda values: str(list(values)))
    refined_out.to_csv(refined_dataset_path, index=False)

    refined_ind = refined[refined["label"] == 1].copy()
    refined_contra = refined[refined["label"] == 0].copy()

    refined_ind_train = _to_training_schema(refined_ind)
    refined_contra_train = _to_training_schema(refined_contra)
    refined_ind_train.to_csv(refined_indications_path, index=False)
    refined_contra_train.to_csv(refined_contraindications_path, index=False)

    report: Dict[str, object] = {
        "total_rows_input": int(total_rows),
        "keep_categories": keep_categories,
        "min_confidence": float(args.min_confidence),
        "drop_needs_review": bool(args.drop_needs_review),
        "rows_after_category_filter": int(total_rows - dropped_by_category),
        "rows_after_confidence_filter": int(total_rows - dropped_by_category - dropped_by_confidence),
        "rows_after_review_filter": int(len(kept)),
        "rows_refined_after_dedup": int(len(refined)),
        "label_counts_refined": {k: int(v) for k, v in refined["label"].value_counts().to_dict().items()},
        "dropped": {
            "category": dropped_by_category,
            "confidence": dropped_by_confidence,
            "needs_review": dropped_by_review,
        },
    }
    utils.save_json(report_path, report)

    print(f"Wrote: {refined_dataset_path}")
    print(f"Wrote: {refined_indications_path}")
    print(f"Wrote: {refined_contraindications_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
