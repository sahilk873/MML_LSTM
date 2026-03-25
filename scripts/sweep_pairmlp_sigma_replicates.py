#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from polypharmacy import data as data_lib
from polypharmacy import model as model_lib
from polypharmacy import utils
from scripts.rank_medic_pairs_rf import (
    DEFAULT_IBD_SUBSTITUTION,
    _load_known_combos_for_target_disease,
    _map_medic_drugs,
    _parse_mondo_codes_from_markdown,
)
from scripts.sweep_pairmlp_sigma import load_base_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-split PairMLP sigma sweeps with replicate training runs and "
            "aggregate mean±SEM for diversity and held-out metrics."
        )
    )
    parser.add_argument(
        "--base-artifact",
        default="artifacts/exp_refined_mixed_twosides_topological512_pairmlp_low_sigma",
    )
    parser.add_argument(
        "--sigmas",
        default="1e-06,5e-06,1e-05,5e-05,1e-04,5e-04,1e-03,5e-03,1e-02,5e-02,1e-01",
    )
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=13)
    parser.add_argument(
        "--output-root",
        default="artifacts/sigma_sweep_topological512_pairmlp_replicates",
    )
    parser.add_argument("--novelty-source", default="filtered", choices=["filtered", "deduped"])
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--ranking-batch-size", type=int, default=200000)
    parser.add_argument(
        "--reuse-singletons",
        action="store_true",
        help="Reuse existing seed-start single-run outputs from prior sweeps when available.",
    )
    return parser.parse_args()


def parse_sigmas(raw: str) -> List[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def sigma_tag(sigma: float) -> str:
    return f"sigma_{sigma:.0e}".replace("+0", "").replace("+", "")


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
    probs = []
    labels = []
    with torch.no_grad():
        for features, batch_labels in loader:
            logits = model(features.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch_labels.numpy())
    return utils.compute_metrics(np.concatenate(labels), np.concatenate(probs))


def train_pair_mlp(
    sigma: float,
    seed: int,
    config: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> Dict[str, object]:
    device = torch.device("cpu")
    local_config = dict(config)
    local_config["seed"] = seed
    utils.set_seeds(seed)

    train_loader = build_loader(
        X_train, y_train, batch_size=int(local_config["pair_mlp_batch_size"]), shuffle=True
    )
    val_loader = build_loader(
        X_val, y_val, batch_size=int(local_config["pair_mlp_batch_size"]), shuffle=False
    )
    test_loader = build_loader(
        X_test, y_test, batch_size=int(local_config["pair_mlp_batch_size"]), shuffle=False
    )
    model = model_lib.PairEmbeddingMLPClassifier(
        input_dim=int(X_train.shape[1]),
        hidden_dim=int(local_config["pair_mlp_hidden_dim"]),
        num_layers=int(local_config["pair_mlp_layers"]),
        dropout=float(local_config["pair_mlp_dropout"]),
        init_sigma=sigma,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(local_config["pair_mlp_learning_rate"]),
        weight_decay=float(local_config["pair_mlp_weight_decay"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    best_val_auc = float("-inf")
    best_epoch = -1
    best_path = output_dir / "pair_mlp_best.pt"

    for epoch in range(1, int(local_config["pair_mlp_epochs"]) + 1):
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
        train_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate_pair_mlp(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
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
                    "hidden_dim": int(local_config["pair_mlp_hidden_dim"]),
                    "num_layers": int(local_config["pair_mlp_layers"]),
                    "dropout": float(local_config["pair_mlp_dropout"]),
                    "init_sigma": sigma,
                    "learning_rate": float(local_config["pair_mlp_learning_rate"]),
                    "weight_decay": float(local_config["pair_mlp_weight_decay"]),
                    "epochs": int(local_config["pair_mlp_epochs"]),
                },
                best_path,
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    best_ckpt = torch.load(best_path, map_location="cpu")
    best_model = model_lib.PairEmbeddingMLPClassifier(
        input_dim=int(best_ckpt["input_dim"]),
        hidden_dim=int(best_ckpt["hidden_dim"]),
        num_layers=int(best_ckpt["num_layers"]),
        dropout=float(best_ckpt["dropout"]),
        init_sigma=None,
    ).to(device)
    best_model.load_state_dict(best_ckpt["model_state"])
    best_val_metrics = evaluate_pair_mlp(best_model, val_loader, device)
    test_metrics = evaluate_pair_mlp(best_model, test_loader, device)
    final_row = history[-1]
    payload = {
        "seed": seed,
        "sigma": sigma,
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_metrics["accuracy"]),
        "best_val_roc_auc": float(best_val_metrics["roc_auc"]),
        "final_train_loss": float(final_row["train_loss"]),
        "final_val_accuracy": float(final_row["val_accuracy"]),
        "heldout_test_accuracy": float(test_metrics["accuracy"]),
        "heldout_test_auroc": float(test_metrics["roc_auc"]),
        "heldout_test_f1": float(test_metrics["f1"]),
        "heldout_test_sensitivity": float(test_metrics["sensitivity"]),
        "heldout_test_specificity": float(test_metrics["specificity"]),
        "heldout_test_tn": int(test_metrics["confusion"]["tn"]),
        "heldout_test_fp": int(test_metrics["confusion"]["fp"]),
        "heldout_test_fn": int(test_metrics["confusion"]["fn"]),
        "heldout_test_tp": int(test_metrics["confusion"]["tp"]),
    }
    with open(output_dir / "replicate_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def pair_flat_index(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    prefix = i * (2 * n - i - 1) // 2
    return prefix + (j - i - 1)


class CachedRanker:
    def __init__(
        self,
        base_artifact: Path,
        novelty_source: str,
        top_n: int,
        batch_size: int,
    ) -> None:
        self.base_artifact = base_artifact
        self.novelty_source = novelty_source
        self.top_n = top_n
        self.batch_size = batch_size

        self.node_ids = np.load("artifacts/precomputed_embeddings/topological/node_ids.npy", allow_pickle=True)
        self.node_embeddings = np.load(
            "artifacts/precomputed_embeddings/topological/embeddings.npy", mmap_mode="r"
        )
        self.disease_rows = _parse_mondo_codes_from_markdown("disease_codes_reference.md")
        matched_df, _ = _map_medic_drugs(
            medic_csv="MeDIC Drug List.csv",
            alias_index_parquet="artifacts/precomputed_embeddings/topological/equivalent_id_to_node_id.parquet",
            allowed_node_ids=None,
        )
        self.selected_drug_ids = sorted(
            matched_df["selected_node_id"].astype(str).unique().tolist()
        )
        self.node_to_idx = {
            str(node_id): idx for idx, node_id in enumerate(self.node_ids)
        }
        self.selected_drug_ids = [
            drug_id for drug_id in self.selected_drug_ids if drug_id in self.node_to_idx
        ]
        self.drug_embeddings = self.node_embeddings[
            np.array([self.node_to_idx[drug_id] for drug_id in self.selected_drug_ids], dtype=np.int64)
        ]
        self.n_drugs = len(self.selected_drug_ids)
        self.pair_i, self.pair_j = np.triu_indices(self.n_drugs, k=1)
        self.num_pairs = int(self.pair_i.shape[0])
        self.drug_to_local = {drug_id: idx for idx, drug_id in enumerate(self.selected_drug_ids)}
        self.exclude_by_disease: Dict[str, np.ndarray] = {}
        self.disease_vec_by_code: Dict[str, np.ndarray] = {}
        self._prepare_disease_cache()

    def _prepare_disease_cache(self) -> None:
        for disease_name, disease_code in self.disease_rows:
            disease_used = DEFAULT_IBD_SUBSTITUTION.get(disease_code, disease_code)
            self.disease_vec_by_code[disease_code] = self.node_embeddings[self.node_to_idx[disease_used]]
            exclude = np.zeros(self.num_pairs, dtype=bool)
            known_combos = _load_known_combos_for_target_disease(
                str(self.base_artifact), self.novelty_source, disease_used
            )
            if known_combos:
                idx_list = []
                for combo in known_combos:
                    if len(combo) != 2:
                        continue
                    drug_a, drug_b = sorted(combo)
                    if drug_a not in self.drug_to_local or drug_b not in self.drug_to_local:
                        continue
                    i = self.drug_to_local[drug_a]
                    j = self.drug_to_local[drug_b]
                    idx_list.append(int(pair_flat_index(np.array([i]), np.array([j]), self.n_drugs)[0]))
                if idx_list:
                    exclude[np.array(idx_list, dtype=np.int64)] = True
            self.exclude_by_disease[disease_code] = exclude

    def _build_score_batch(self, model_path: Path):
        checkpoint = torch.load(model_path, map_location="cpu")
        model = model_lib.PairEmbeddingMLPClassifier(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_layers=int(checkpoint["num_layers"]),
            dropout=float(checkpoint["dropout"]),
            init_sigma=None,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        def score_batch(X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                logits = model(torch.from_numpy(X))
                return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

        return score_batch

    def rank_model(self, model_path: Path) -> Dict[str, object]:
        score_batch = self._build_score_batch(model_path)
        disease_rows = []
        combined_pairs = []
        feat_dim = self.drug_embeddings.shape[1] * 2 + self.drug_embeddings.shape[1]

        for disease_name, disease_code in self.disease_rows:
            exclude = self.exclude_by_disease[disease_code]
            disease_vec = self.disease_vec_by_code[disease_code]
            top_scores = np.empty(0, dtype=np.float32)
            top_pair_idx = np.empty(0, dtype=np.int64)

            for start in range(0, self.num_pairs, self.batch_size):
                end = min(self.num_pairs, start + self.batch_size)
                valid = ~exclude[start:end]
                if not np.any(valid):
                    continue
                ii = self.pair_i[start:end][valid]
                jj = self.pair_j[start:end][valid]
                n_rows = ii.shape[0]
                X = np.empty((n_rows, feat_dim), dtype=np.float32)
                emb_dim = self.drug_embeddings.shape[1]
                X[:, :emb_dim] = self.drug_embeddings[ii]
                X[:, emb_dim : 2 * emb_dim] = self.drug_embeddings[jj]
                X[:, 2 * emb_dim :] = disease_vec
                scores = score_batch(X)
                pair_idx = np.arange(start, end, dtype=np.int64)[valid]

                if top_scores.size == 0:
                    merged_scores = scores
                    merged_idx = pair_idx
                else:
                    merged_scores = np.concatenate([top_scores, scores])
                    merged_idx = np.concatenate([top_pair_idx, pair_idx])

                if merged_scores.size > self.top_n:
                    keep = np.argpartition(-merged_scores, self.top_n - 1)[: self.top_n]
                    top_scores = merged_scores[keep]
                    top_pair_idx = merged_idx[keep]
                else:
                    top_scores = merged_scores
                    top_pair_idx = merged_idx

            order = np.argsort(-top_scores)
            top_scores = top_scores[order]
            top_pair_idx = top_pair_idx[order]
            drug_ids_1 = [self.selected_drug_ids[int(self.pair_i[idx])] for idx in top_pair_idx]
            drug_ids_2 = [self.selected_drug_ids[int(self.pair_j[idx])] for idx in top_pair_idx]
            unique_drugs = len(set(drug_ids_1).union(drug_ids_2))
            unique_pairs = len({tuple(sorted((a, b))) for a, b in zip(drug_ids_1, drug_ids_2)})
            disease_rows.append(
                {
                    "disease_name": disease_name,
                    "unique_drugs_in_top50": int(unique_drugs),
                    "unique_pairs_in_top50": int(unique_pairs),
                }
            )
            combined_pairs.extend(tuple(sorted((a, b))) for a, b in zip(drug_ids_1, drug_ids_2))

        pair_counts = pd.Series(combined_pairs).value_counts()
        return {
            "mean_unique_drugs_in_top50": float(np.mean([row["unique_drugs_in_top50"] for row in disease_rows])),
            "min_unique_drugs_in_top50": int(np.min([row["unique_drugs_in_top50"] for row in disease_rows])),
            "max_unique_drugs_in_top50": int(np.max([row["unique_drugs_in_top50"] for row in disease_rows])),
            "global_unique_pairs": int(pair_counts.shape[0]),
            "pairs_recommended_for_multiple_diseases": int((pair_counts > 1).sum()),
            "reused_pair_rows": int(pair_counts[pair_counts > 1].sum()),
            "per_disease": disease_rows,
        }


def mean_sem(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.array(list(values), dtype=float)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean, 0.0
    sem = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    return mean, sem


def aggregate_sigma_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"n_replicates": len(rows)}
    scalar_metrics = [
        "mean_unique_drugs_in_top50",
        "global_unique_pairs",
        "pairs_recommended_for_multiple_diseases",
        "reused_pair_rows",
        "heldout_test_auroc",
        "heldout_test_accuracy",
        "heldout_test_f1",
        "heldout_test_sensitivity",
        "heldout_test_specificity",
        "best_val_accuracy",
        "best_val_roc_auc",
        "final_train_loss",
        "final_val_accuracy",
    ]
    for key in scalar_metrics:
        mean, sem = mean_sem(float(row[key]) for row in rows)
        out[f"{key}_mean"] = mean
        out[f"{key}_sem"] = sem
    for key in ["heldout_test_tn", "heldout_test_fp", "heldout_test_fn", "heldout_test_tp"]:
        mean, sem = mean_sem(float(row[key]) for row in rows)
        out[f"{key}_mean"] = mean
        out[f"{key}_sem"] = sem
    return out


def existing_singleton_lookup() -> Dict[float, Dict[str, object]]:
    lookup: Dict[float, Dict[str, object]] = {}
    candidate_roots = [
        Path("artifacts/sigma_sweep_topological512_pairmlp_v2"),
        Path("artifacts/sigma_sweep_topological512_pairmlp_v3"),
    ]
    for root in candidate_roots:
        summary_path = root / "sigma_sweep_summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        for row in df.to_dict(orient="records"):
            sigma = float(row["sigma"])
            tag = sigma_tag(sigma)
            metrics_path = root / tag / "metrics_summary.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
                lookup[sigma] = {
                    "metrics": row,
                    "metrics_summary": metrics,
                }
    return lookup


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sigmas = parse_sigmas(args.sigmas)
    seeds = [args.seed_start + i for i in range(args.replicates)]
    base_artifact = Path(args.base_artifact)

    base_state = load_base_state(base_artifact)
    config = dict(base_state["config"])
    filtered_df = base_state["filtered_df"]
    splits = base_state["splits"]
    train_df = filtered_df.iloc[splits["train_idx"]].copy()
    val_df = filtered_df.iloc[splits["val_idx"]].copy()
    test_df = filtered_df.iloc[splits["test_idx"]].copy()

    from experiment import build_rf_features

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

    ranker = CachedRanker(
        base_artifact=base_artifact,
        novelty_source=args.novelty_source,
        top_n=args.top_n,
        batch_size=args.ranking_batch_size,
    )
    singleton_lookup = existing_singleton_lookup() if args.reuse_singletons else {}

    aggregate_rows = []
    per_replicate_rows = []
    disease_rows = []

    for sigma in sigmas:
        tag = sigma_tag(sigma)
        sigma_dir = output_root / tag
        sigma_dir.mkdir(parents=True, exist_ok=True)
        sigma_rows = []

        for rep_idx, seed in enumerate(seeds):
            rep_dir = sigma_dir / f"rep_{rep_idx:02d}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            start = time.time()

            if rep_idx == 0 and sigma in singleton_lookup:
                source_metrics = singleton_lookup[sigma]["metrics"]
                row = {
                    "seed": seed,
                    "sigma": sigma,
                    "best_epoch": None,
                    "best_val_accuracy": float(source_metrics["best_val_accuracy"]),
                    "best_val_roc_auc": float(source_metrics["best_val_roc_auc"]),
                    "final_train_loss": float(source_metrics["final_train_loss"]),
                    "final_val_accuracy": float(source_metrics["final_val_accuracy"]),
                    "heldout_test_accuracy": float(source_metrics["heldout_test_accuracy"]),
                    "heldout_test_auroc": float(source_metrics["heldout_test_auroc"]),
                    "heldout_test_f1": float(source_metrics["heldout_test_f1"]),
                    "heldout_test_sensitivity": float(source_metrics["heldout_test_sensitivity"]),
                    "heldout_test_specificity": float(source_metrics["heldout_test_specificity"]),
                    "heldout_test_tn": int(source_metrics["heldout_test_tn"]),
                    "heldout_test_fp": int(source_metrics["heldout_test_fp"]),
                    "heldout_test_fn": int(source_metrics["heldout_test_fn"]),
                    "heldout_test_tp": int(source_metrics["heldout_test_tp"]),
                    "mean_unique_drugs_in_top50": float(source_metrics["mean_unique_drugs_in_top50"]),
                    "global_unique_pairs": int(source_metrics["global_unique_pairs"]),
                    "pairs_recommended_for_multiple_diseases": int(source_metrics["pairs_recommended_for_multiple_diseases"]),
                    "reused_pair_rows": int(source_metrics["reused_pair_rows"]),
                    "timing_seconds": 0.0,
                    "reused_existing": True,
                }
                existing_metrics = singleton_lookup[sigma]["metrics_summary"]
                if "best_epoch" in existing_metrics:
                    row["best_epoch"] = existing_metrics["best_epoch"]
                row["per_disease"] = []
                prior_root = (
                    Path("artifacts/sigma_sweep_topological512_pairmlp_v2")
                    if (Path("artifacts/sigma_sweep_topological512_pairmlp_v2") / tag).exists()
                    else Path("artifacts/sigma_sweep_topological512_pairmlp_v3")
                )
                per_disease_path = prior_root / "sigma_sweep_per_disease.csv"
                if per_disease_path.exists():
                    per_disease_df = pd.read_csv(per_disease_path)
                    row["per_disease"] = per_disease_df[per_disease_df["sigma"] == sigma].to_dict(orient="records")
            else:
                train_metrics = train_pair_mlp(
                    sigma=sigma,
                    seed=seed,
                    config=config,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    output_dir=rep_dir,
                )
                ranking_metrics = ranker.rank_model(rep_dir / "pair_mlp_best.pt")
                row = {
                    **train_metrics,
                    **{k: v for k, v in ranking_metrics.items() if k != "per_disease"},
                    "timing_seconds": float(time.time() - start),
                    "reused_existing": False,
                    "per_disease": ranking_metrics["per_disease"],
                }
                with open(rep_dir / "ranking_summary.json", "w", encoding="utf-8") as handle:
                    json.dump(ranking_metrics, handle, indent=2)

            sigma_rows.append(row)
            per_replicate_rows.append({k: v for k, v in row.items() if k != "per_disease"})
            for disease_row in row["per_disease"]:
                disease_rows.append(
                    {
                        "sigma": sigma,
                        "seed": seed,
                        **disease_row,
                    }
                )

        aggregate = aggregate_sigma_rows(sigma_rows)
        aggregate["sigma"] = sigma
        aggregate_rows.append(aggregate)

    pd.DataFrame(per_replicate_rows).sort_values(["sigma", "seed"]).to_csv(
        output_root / "sigma_replicates_per_run.csv", index=False
    )
    pd.DataFrame(aggregate_rows).sort_values("sigma").to_csv(
        output_root / "sigma_replicates_summary.csv", index=False
    )
    pd.DataFrame(disease_rows).sort_values(["sigma", "seed", "disease_name"]).to_csv(
        output_root / "sigma_replicates_per_disease.csv", index=False
    )
    payload = {
        "base_artifact": str(base_artifact),
        "novelty_source": args.novelty_source,
        "sigmas": sigmas,
        "seeds": seeds,
        "summary": aggregate_rows,
    }
    with open(output_root / "sigma_replicates_report.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
