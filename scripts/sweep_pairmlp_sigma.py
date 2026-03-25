#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiment import build_rf_features
from polypharmacy import data as data_lib
from polypharmacy import model as model_lib
from polypharmacy import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep PairMLP initialization sigma on a fixed artifact split and "
            "quantify repurposing diversity."
        )
    )
    parser.add_argument(
        "--base-artifact",
        default="artifacts/exp_refined_mixed_twosides_topological512_pairmlp_low_sigma",
    )
    parser.add_argument(
        "--sigmas",
        default="5e-05,1e-04,5e-04,1e-03,5e-03,1e-02",
        help="Comma-separated sigma values for PairMLP init.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/sigma_sweep_topological512_pairmlp",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--ranking-batch-size", type=int, default=200000)
    parser.add_argument("--ranking-max-workers", type=int, default=4)
    parser.add_argument("--novelty-source", default="deduped", choices=["deduped", "filtered"])
    parser.add_argument(
        "--rf-ranking-summary",
        default="artifacts/rf_repurpose_top50/run_20260306_035906/summary.json",
    )
    parser.add_argument(
        "--rf-ranking-top50",
        default="artifacts/rf_repurpose_top50/run_20260306_035906/top50_all_diseases.csv",
    )
    return parser.parse_args()


def parse_sigmas(raw: str) -> List[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("At least one sigma value is required.")
    return values


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_base_state(base_artifact: Path) -> Dict[str, object]:
    config = utils.load_json(str(base_artifact / "config.json"))
    filtered_df = pd.read_csv(base_artifact / "filtered_dataset_run.csv")
    filtered_df["drug_set"] = filtered_df["drug_set"].apply(data_lib.parse_list_column)
    splits = np.load(base_artifact / "splits.npz")
    drug_embeddings = np.load(base_artifact / "drug_embeddings.npy")
    disease_embeddings = np.load(base_artifact / "disease_embeddings.npy")
    drug_ids = utils.load_json(str(base_artifact / "drug_vocab.json"))["ids"]
    disease_ids = utils.load_json(str(base_artifact / "disease_vocab.json"))["ids"]
    drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids) if idx != 0}
    disease_to_idx = {disease_id: idx for idx, disease_id in enumerate(disease_ids)}
    return {
        "config": config,
        "filtered_df": filtered_df,
        "splits": splits,
        "drug_embeddings": drug_embeddings,
        "disease_embeddings": disease_embeddings,
        "drug_to_idx": drug_to_idx,
        "disease_to_idx": disease_to_idx,
    }


def dataframe_from_split(filtered_df: pd.DataFrame, split_positions: np.ndarray) -> pd.DataFrame:
    if split_positions.ndim != 1:
        raise ValueError("Expected 1-D split index array.")
    return filtered_df.iloc[split_positions].copy()


def build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_pair_mlp(
    model: model_lib.PairEmbeddingMLPClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    if not all_probs:
        return {"roc_auc": float("nan"), "accuracy": float("nan")}
    return utils.compute_metrics(np.concatenate(all_labels), np.concatenate(all_probs))


def train_pair_mlp_for_sigma(
    sigma: float,
    config: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    utils.set_seeds(int(config["seed"]))
    train_loader = build_loader(
        X_train, y_train, batch_size=int(config["pair_mlp_batch_size"]), shuffle=True
    )
    val_loader = build_loader(
        X_val, y_val, batch_size=int(config["pair_mlp_batch_size"]), shuffle=False
    )
    test_loader = build_loader(
        X_test, y_test, batch_size=int(config["pair_mlp_batch_size"]), shuffle=False
    )
    model = model_lib.PairEmbeddingMLPClassifier(
        input_dim=int(X_train.shape[1]),
        hidden_dim=int(config["pair_mlp_hidden_dim"]),
        num_layers=int(config["pair_mlp_layers"]),
        dropout=float(config["pair_mlp_dropout"]),
        init_sigma=sigma,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["pair_mlp_learning_rate"]),
        weight_decay=float(config["pair_mlp_weight_decay"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    history: List[Dict[str, object]] = []
    best_val_auc = float("-inf")
    best_epoch = -1
    best_checkpoint_path = output_dir / "pair_mlp_best.pt"
    last_train_loss = float("nan")

    for epoch in range(1, int(config["pair_mlp_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))
        last_train_loss = avg_loss
        val_metrics = evaluate_pair_mlp(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_roc_auc": float(val_metrics["roc_auc"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_sensitivity": float(val_metrics["sensitivity"]),
                "val_specificity": float(val_metrics["specificity"]),
                "val_f1": float(val_metrics["f1"]),
            }
        )
        if not np.isnan(val_metrics["roc_auc"]) and float(val_metrics["roc_auc"]) > best_val_auc:
            best_val_auc = float(val_metrics["roc_auc"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": int(X_train.shape[1]),
                    "hidden_dim": int(config["pair_mlp_hidden_dim"]),
                    "num_layers": int(config["pair_mlp_layers"]),
                    "dropout": float(config["pair_mlp_dropout"]),
                    "init_sigma": sigma,
                    "learning_rate": float(config["pair_mlp_learning_rate"]),
                    "weight_decay": float(config["pair_mlp_weight_decay"]),
                    "epochs": int(config["pair_mlp_epochs"]),
                },
                best_checkpoint_path,
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
    best_model = model_lib.PairEmbeddingMLPClassifier(
        input_dim=int(best_checkpoint["input_dim"]),
        hidden_dim=int(best_checkpoint["hidden_dim"]),
        num_layers=int(best_checkpoint["num_layers"]),
        dropout=float(best_checkpoint["dropout"]),
        init_sigma=None,
    ).to(device)
    best_model.load_state_dict(best_checkpoint["model_state"])

    best_val_metrics = evaluate_pair_mlp(best_model, val_loader, device)
    test_metrics = evaluate_pair_mlp(best_model, test_loader, device)
    final_val_metrics = history[-1] if history else {}
    summary = {
        "sigma": sigma,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint_path),
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "final_epoch": int(history[-1]["epoch"]) if history else None,
        "final_train_loss": last_train_loss,
        "final_val_accuracy": float(final_val_metrics.get("val_accuracy", float("nan"))),
        "final_val_roc_auc": float(final_val_metrics.get("val_roc_auc", float("nan"))),
        "best_val_accuracy": float(best_val_metrics["accuracy"]),
        "best_val_roc_auc": float(best_val_metrics["roc_auc"]),
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_ranking(
    base_artifact: Path,
    model_path: Path,
    ranking_dir: Path,
    top_n: int,
    batch_size: int,
    max_workers: int,
    novelty_source: str,
) -> None:
    ranking_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(Path(".venv/bin/python")),
        "scripts/rank_medic_pairs_rf.py",
        "--model-type",
        "pair_mlp",
        "--model-output-dir",
        str(base_artifact),
        "--model-path",
        str(model_path),
        "--precomputed-node-ids",
        "artifacts/precomputed_embeddings/topological/node_ids.npy",
        "--precomputed-embeddings",
        "artifacts/precomputed_embeddings/topological/embeddings.npy",
        "--medic-drug-list",
        "MeDIC Drug List.csv",
        "--alias-index",
        "artifacts/precomputed_embeddings/topological/equivalent_id_to_node_id.parquet",
        "--disease-reference-md",
        "disease_codes_reference.md",
        "--novelty-source",
        novelty_source,
        "--top-n",
        str(top_n),
        "--batch-size",
        str(batch_size),
        "--max-workers",
        str(max_workers),
        "--output-dir",
        str(ranking_dir.parent),
        "--run-name",
        ranking_dir.name,
    ]
    subprocess.run(cmd, check=True)


def summarize_diversity(top50_csv: Path) -> Dict[str, object]:
    df = pd.read_csv(top50_csv)
    rows = []
    for disease_name, group in df.groupby("disease_name"):
        drugs = set(group["drug_id_1"]).union(set(group["drug_id_2"]))
        rows.append(
            {
                "disease_name": disease_name,
                "rows": int(len(group)),
                "unique_drugs_in_top50": int(len(drugs)),
                "unique_pairs_in_top50": int(
                    len({tuple(sorted((a, b))) for a, b in zip(group["drug_id_1"], group["drug_id_2"])})
                ),
            }
        )
    disease_df = pd.DataFrame(rows).sort_values("disease_name").reset_index(drop=True)
    summary = {
        "per_disease": disease_df.to_dict(orient="records"),
        "mean_unique_drugs_in_top50": float(disease_df["unique_drugs_in_top50"].mean()),
        "min_unique_drugs_in_top50": int(disease_df["unique_drugs_in_top50"].min()),
        "max_unique_drugs_in_top50": int(disease_df["unique_drugs_in_top50"].max()),
        "mean_unique_pairs_in_top50": float(disease_df["unique_pairs_in_top50"].mean()),
    }
    return summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    sigmas = parse_sigmas(args.sigmas)
    base_artifact = Path(args.base_artifact)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    base_state = load_base_state(base_artifact)
    config = dict(base_state["config"])
    config["seed"] = args.seed

    filtered_df = base_state["filtered_df"]
    splits = base_state["splits"]
    train_df = dataframe_from_split(filtered_df, splits["train_idx"])
    val_df = dataframe_from_split(filtered_df, splits["val_idx"])
    test_df = dataframe_from_split(filtered_df, splits["test_idx"])

    X_train, y_train = build_rf_features(
        train_df,
        base_state["drug_to_idx"],
        base_state["disease_to_idx"],
        base_state["drug_embeddings"],
        base_state["disease_embeddings"],
    )
    X_val, y_val = build_rf_features(
        val_df,
        base_state["drug_to_idx"],
        base_state["disease_to_idx"],
        base_state["drug_embeddings"],
        base_state["disease_embeddings"],
    )
    X_test, y_test = build_rf_features(
        test_df,
        base_state["drug_to_idx"],
        base_state["disease_to_idx"],
        base_state["drug_embeddings"],
        base_state["disease_embeddings"],
    )

    rf_metrics = utils.load_json(str(base_artifact / "metrics.json")).get("rf_metrics", {})
    rf_ranking_summary = utils.load_json(args.rf_ranking_summary)
    rf_diversity = summarize_diversity(Path(args.rf_ranking_top50))

    sweep_rows: List[Dict[str, object]] = []
    disease_rows: List[Dict[str, object]] = []
    for sigma in sigmas:
        sigma_tag = f"sigma_{sigma:.0e}".replace("+0", "").replace("+", "")
        sigma_dir = output_root / sigma_tag
        sigma_dir.mkdir(parents=True, exist_ok=True)
        metrics = train_pair_mlp_for_sigma(
            sigma=sigma,
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            device=device,
            output_dir=sigma_dir,
        )
        ranking_parent = output_root / "rankings"
        ranking_dir = ranking_parent / sigma_tag
        run_ranking(
            base_artifact=base_artifact,
            model_path=sigma_dir / "pair_mlp_best.pt",
            ranking_dir=ranking_dir,
            top_n=args.top_n,
            batch_size=args.ranking_batch_size,
            max_workers=args.ranking_max_workers,
            novelty_source=args.novelty_source,
        )
        diversity = summarize_diversity(ranking_dir / "top50_all_diseases.csv")
        for disease_entry in diversity["per_disease"]:
            disease_rows.append(
                {
                    "sigma": sigma,
                    **disease_entry,
                }
            )
        ranking_summary_path = ranking_dir / "summary.json"
        ranking_summary = utils.load_json(str(ranking_summary_path)) if ranking_summary_path.exists() else {}
        test_confusion = metrics["test_metrics"].get("confusion", {})
        sweep_rows.append(
            {
                "sigma": sigma,
                "best_epoch": metrics["best_epoch"],
                "final_train_loss": metrics["final_train_loss"],
                "final_val_accuracy": metrics["final_val_accuracy"],
                "best_val_accuracy": metrics["best_val_accuracy"],
                "best_val_roc_auc": metrics["best_val_roc_auc"],
                "heldout_test_auroc": metrics["test_metrics"]["roc_auc"],
                "heldout_test_accuracy": metrics["test_metrics"]["accuracy"],
                "heldout_test_f1": metrics["test_metrics"]["f1"],
                "heldout_test_sensitivity": metrics["test_metrics"]["sensitivity"],
                "heldout_test_specificity": metrics["test_metrics"]["specificity"],
                "heldout_test_tn": test_confusion.get("tn"),
                "heldout_test_fp": test_confusion.get("fp"),
                "heldout_test_fn": test_confusion.get("fn"),
                "heldout_test_tp": test_confusion.get("tp"),
                "mean_unique_drugs_in_top50": diversity["mean_unique_drugs_in_top50"],
                "min_unique_drugs_in_top50": diversity["min_unique_drugs_in_top50"],
                "max_unique_drugs_in_top50": diversity["max_unique_drugs_in_top50"],
                "mean_unique_pairs_in_top50": diversity["mean_unique_pairs_in_top50"],
                "global_unique_pairs": ranking_summary.get("uniqueness_summary", {}).get("unique_pairs"),
                "pairs_recommended_for_multiple_diseases": ranking_summary.get("uniqueness_summary", {}).get(
                    "pairs_recommended_for_multiple_diseases"
                ),
                "reused_pair_rows": ranking_summary.get("uniqueness_summary", {}).get("reused_pair_rows"),
            }
        )

    write_csv(output_root / "sigma_sweep_summary.csv", sweep_rows)
    write_csv(output_root / "sigma_sweep_per_disease.csv", disease_rows)
    report = {
        "base_artifact": str(base_artifact),
        "device": str(device),
        "seed": args.seed,
        "novelty_source": args.novelty_source,
        "sigmas": sigmas,
        "fixed_pair_mlp_config": {
            key: config[key]
            for key in [
                "pair_mlp_hidden_dim",
                "pair_mlp_layers",
                "pair_mlp_dropout",
                "pair_mlp_epochs",
                "pair_mlp_batch_size",
                "pair_mlp_learning_rate",
                "pair_mlp_weight_decay",
            ]
        },
        "heldout_test_note": (
            "This repository does not define a separate external validation set for PairMLP. "
            "Reported AUROC/confusion use the fixed held-out 2-drug test split from splits.npz."
        ),
        "rf_baseline": {
            "metrics": rf_metrics,
            "ranking_summary": rf_ranking_summary,
            "diversity": rf_diversity,
        },
        "pair_mlp_sweep": sweep_rows,
    }
    with open(output_root / "sigma_sweep_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
