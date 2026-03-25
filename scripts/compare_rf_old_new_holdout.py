#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from polypharmacy import data as data_lib
from polypharmacy import kg as kg_lib
from polypharmacy import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RF performance on old vs new ground truth using one shared holdout."
        )
    )
    parser.add_argument("--old-indications", default="indications_norm_dedup.csv")
    parser.add_argument("--old-contraindications", default="contraindications_norm_dedup.csv")
    parser.add_argument(
        "--new-indications", default="artifacts/refined_gt/refined_indications.csv"
    )
    parser.add_argument("--new-contraindications", default="contraindications_norm_dedup.csv")
    parser.add_argument(
        "--new-twosides",
        default="twosides_ddi_prefixed_normalized.csv",
        help="Optional TWOSIDES negatives used only for the new-GT dataset.",
    )
    parser.add_argument(
        "--kg-embeddings",
        default="artifacts/precomputed_embeddings/topological/embeddings.npy",
    )
    parser.add_argument(
        "--kg-embedding-ids",
        default="artifacts/precomputed_embeddings/topological/node_ids.npy",
    )
    parser.add_argument(
        "--alias-index",
        default="artifacts/precomputed_embeddings/topological/equivalent_id_to_node_id.parquet",
        help="Alias-to-canonical node ID parquet used to resolve equivalent IDs everywhere.",
    )
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="artifacts/rf_old_vs_new_holdout")
    return parser.parse_args()


def _canonical_key(drug_set: Sequence[str], disease_id: str) -> str:
    drugs = tuple(sorted(str(drug_id) for drug_id in drug_set))
    return "|".join([drugs[0], drugs[1], str(disease_id)])


def _prepare_df(
    indications_path: str,
    contraindications_path: str,
    kg_nodes: Iterable[str],
    enable_mixed_negatives: bool,
    twosides_path: str | None,
    alias_index_path: str,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]]:
    dedupe_report: Dict[str, object] = {}
    deduped_df, conflict_count = data_lib.load_deduped_dataframe(
        indications_path=indications_path,
        contraindications_path=contraindications_path,
        twosides_contraindications_path=twosides_path,
        alias_index_path=alias_index_path,
        enable_mixed_negatives=enable_mixed_negatives,
        random_negative_ratio=1.0,
        random_negative_strategy="disease_shuffle",
        seed=seed,
        report_out=dedupe_report,
    )
    filtered_df, dropped_df, drop_stats = data_lib.filter_by_kg_coverage(deduped_df, kg_nodes)
    two_drug_df = filtered_df[filtered_df["drug_set"].apply(lambda ds: len(ds) == 2)].copy()
    two_drug_df["pair_key"] = two_drug_df.apply(
        lambda row: _canonical_key(row.drug_set, row.condition_id_norm), axis=1
    )
    report = {
        "conflict_count": int(conflict_count),
        "deduped_rows": int(len(deduped_df)),
        "filtered_rows": int(len(filtered_df)),
        "two_drug_rows": int(len(two_drug_df)),
        "dropped_rows": int(len(dropped_df)),
        "drop_stats": drop_stats,
        "dedupe_report": dedupe_report,
    }
    return two_drug_df, report, drop_stats


def _choose_holdout_keys(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    holdout_frac: float,
    seed: int,
) -> pd.DataFrame:
    old_view = old_df[["pair_key", "label"]].rename(columns={"label": "old_label"})
    new_view = new_df[["pair_key", "label"]].rename(columns={"label": "new_label"})
    common = old_view.merge(new_view, on="pair_key", how="inner").drop_duplicates("pair_key")
    if common.empty:
        raise ValueError("No common pair keys between old and new datasets after filtering.")
    common["label_pair"] = common.apply(
        lambda row: f"{int(row.old_label)}->{int(row.new_label)}", axis=1
    )
    if len(common) < 2:
        raise ValueError("Need at least 2 common keys to create a holdout.")
    holdout_n = max(1, int(len(common) * holdout_frac))
    holdout_n = min(holdout_n, len(common) - 1)
    stratify = common["label_pair"]
    if stratify.nunique() > 1 and stratify.value_counts().min() >= 2 and holdout_n >= stratify.nunique():
        _, holdout = train_test_split(
            common,
            test_size=holdout_n,
            random_state=seed,
            stratify=stratify,
        )
    else:
        holdout = common.sample(n=holdout_n, random_state=seed)
    return holdout.sort_values("pair_key").reset_index(drop=True)


def _build_features(
    df: pd.DataFrame,
    node_to_idx: Dict[str, int],
    node_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    labels: List[int] = []
    for row in df.itertuples(index=False):
        drug_ids = list(row.drug_set)
        disease_id = str(row.condition_id_norm)
        d1, d2 = drug_ids
        feature = np.concatenate(
            (
                np.asarray(node_embeddings[node_to_idx[str(d1)]], dtype=np.float32),
                np.asarray(node_embeddings[node_to_idx[str(d2)]], dtype=np.float32),
                np.asarray(node_embeddings[node_to_idx[disease_id]], dtype=np.float32),
            )
        )
        rows.append(feature)
        labels.append(int(row.label))
    if not rows:
        raise ValueError("RF split contains no rows.")
    return np.stack(rows, axis=0), np.asarray(labels, dtype=np.int64)


def _compute_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, object]:
    preds = (probs >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "sensitivity": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _fit_and_eval_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    node_to_idx: Dict[str, int],
    node_embeddings: np.ndarray,
    estimators: int,
    max_depth: int,
    seed: int,
) -> Tuple[RandomForestClassifier, Dict[str, object]]:
    X_train, y_train = _build_features(train_df, node_to_idx, node_embeddings)
    X_test, y_test = _build_features(test_df, node_to_idx, node_embeddings)
    clf = RandomForestClassifier(
        n_estimators=estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    metrics = _compute_metrics(y_test, probs)
    metrics["train_examples"] = int(len(y_train))
    metrics["test_examples"] = int(len(y_test))
    return clf, metrics


def _describe_label_distribution(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    counts = df[label_col].value_counts().to_dict()
    return {str(int(key)): int(value) for key, value in counts.items()}


def main() -> None:
    args = parse_args()
    utils.set_seeds(args.seed)
    utils.ensure_dir(args.output_dir)

    node_ids, node_embeddings = kg_lib.load_precomputed_embeddings(
        args.kg_embeddings, args.kg_embedding_ids
    )
    kg_nodes = set(node_ids)
    node_to_idx = {str(node_id): idx for idx, node_id in enumerate(node_ids)}

    old_df, old_report, _ = _prepare_df(
        indications_path=args.old_indications,
        contraindications_path=args.old_contraindications,
        kg_nodes=kg_nodes,
        enable_mixed_negatives=True,
        twosides_path=None,
        alias_index_path=args.alias_index,
        seed=args.seed,
    )
    new_df, new_report, _ = _prepare_df(
        indications_path=args.new_indications,
        contraindications_path=args.new_contraindications,
        kg_nodes=kg_nodes,
        enable_mixed_negatives=True,
        twosides_path=args.new_twosides,
        alias_index_path=args.alias_index,
        seed=args.seed,
    )

    holdout_df = _choose_holdout_keys(old_df, new_df, args.holdout_frac, args.seed)
    holdout_keys = set(holdout_df["pair_key"])

    old_train = old_df[~old_df["pair_key"].isin(holdout_keys)].copy()
    old_test = old_df[old_df["pair_key"].isin(holdout_keys)].copy()
    new_train = new_df[~new_df["pair_key"].isin(holdout_keys)].copy()
    new_test = new_df[new_df["pair_key"].isin(holdout_keys)].copy()

    old_model, old_metrics = _fit_and_eval_rf(
        old_train,
        old_test,
        node_to_idx=node_to_idx,
        node_embeddings=node_embeddings,
        estimators=args.rf_estimators,
        max_depth=args.rf_max_depth,
        seed=args.seed,
    )
    new_model, new_metrics = _fit_and_eval_rf(
        new_train,
        new_test,
        node_to_idx=node_to_idx,
        node_embeddings=node_embeddings,
        estimators=args.rf_estimators,
        max_depth=args.rf_max_depth,
        seed=args.seed,
    )

    import pickle

    with open(os.path.join(args.output_dir, "rf_old_gt.pkl"), "wb") as handle:
        pickle.dump(old_model, handle)
    with open(os.path.join(args.output_dir, "rf_new_gt.pkl"), "wb") as handle:
        pickle.dump(new_model, handle)

    holdout_export = holdout_df.copy()
    holdout_export[["drug_id_1", "drug_id_2", "condition_id_norm"]] = holdout_export["pair_key"].str.split(
        "|", expand=True
    )
    holdout_export.to_csv(os.path.join(args.output_dir, "shared_holdout.csv"), index=False)

    comparison = pd.DataFrame(
        [
            {
                "model": "rf_old_gt",
                "roc_auc": old_metrics["roc_auc"],
                "accuracy": old_metrics["accuracy"],
                "precision": old_metrics["precision"],
                "sensitivity": old_metrics["sensitivity"],
                "specificity": old_metrics["specificity"],
                "f1": old_metrics["f1"],
                "tn": old_metrics["confusion"]["tn"],
                "fp": old_metrics["confusion"]["fp"],
                "fn": old_metrics["confusion"]["fn"],
                "tp": old_metrics["confusion"]["tp"],
                "train_examples": old_metrics["train_examples"],
                "test_examples": old_metrics["test_examples"],
            },
            {
                "model": "rf_new_gt",
                "roc_auc": new_metrics["roc_auc"],
                "accuracy": new_metrics["accuracy"],
                "precision": new_metrics["precision"],
                "sensitivity": new_metrics["sensitivity"],
                "specificity": new_metrics["specificity"],
                "f1": new_metrics["f1"],
                "tn": new_metrics["confusion"]["tn"],
                "fp": new_metrics["confusion"]["fp"],
                "fn": new_metrics["confusion"]["fn"],
                "tp": new_metrics["confusion"]["tp"],
                "train_examples": new_metrics["train_examples"],
                "test_examples": new_metrics["test_examples"],
            },
        ]
    )
    comparison.to_csv(os.path.join(args.output_dir, "rf_comparison.csv"), index=False)

    summary = {
        "seed": args.seed,
        "holdout_frac": float(args.holdout_frac),
        "rf_estimators": int(args.rf_estimators),
        "rf_max_depth": int(args.rf_max_depth),
        "old_dataset": old_report,
        "new_dataset": new_report,
        "shared_holdout": {
            "num_common_keys": int(
                len(set(old_df["pair_key"]).intersection(set(new_df["pair_key"])))
            ),
            "num_holdout_keys": int(len(holdout_df)),
            "holdout_label_pair_counts": {
                str(key): int(value)
                for key, value in holdout_df["label_pair"].value_counts().to_dict().items()
            },
            "old_label_counts": _describe_label_distribution(holdout_df, "old_label"),
            "new_label_counts": _describe_label_distribution(holdout_df, "new_label"),
        },
        "rf_old_gt": old_metrics,
        "rf_new_gt": new_metrics,
        "delta_new_minus_old": {
            metric: float(new_metrics[metric]) - float(old_metrics[metric])
            for metric in ["roc_auc", "accuracy", "precision", "sensitivity", "specificity", "f1"]
        },
    }
    utils.save_json(os.path.join(args.output_dir, "summary.json"), summary)

    print(f"Wrote comparison directory: {args.output_dir}")
    print(
        "Shared holdout | "
        f"common_keys={summary['shared_holdout']['num_common_keys']} "
        f"holdout_keys={summary['shared_holdout']['num_holdout_keys']} "
        f"label_pairs={summary['shared_holdout']['holdout_label_pair_counts']}"
    )
    print(
        "RF old GT | "
        f"auc={old_metrics['roc_auc']:.4f} acc={old_metrics['accuracy']:.4f} "
        f"precision={old_metrics['precision']:.4f} f1={old_metrics['f1']:.4f} "
        f"confusion={old_metrics['confusion']}"
    )
    print(
        "RF new GT | "
        f"auc={new_metrics['roc_auc']:.4f} acc={new_metrics['accuracy']:.4f} "
        f"precision={new_metrics['precision']:.4f} f1={new_metrics['f1']:.4f} "
        f"confusion={new_metrics['confusion']}"
    )


if __name__ == "__main__":
    main()
