import argparse
import itertools
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from polypharmacy import config as config_lib
from polypharmacy import data as data_lib
from polypharmacy import kg as kg_lib
from polypharmacy import model as model_lib
from polypharmacy import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RF vs LSTM comparison.")
    parser.add_argument("--config", default=None, help="Optional JSON config overrides.")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--indications", default="indications_norm.csv")
    parser.add_argument("--contraindications", default="contraindications_norm.csv")
    parser.add_argument("--kg", default="kg_edges.parquet")
    parser.add_argument(
        "--kg-embeddings",
        default=None,
        help="Optional precomputed KG embeddings (.npz or .npy).",
    )
    parser.add_argument(
        "--kg-embedding-ids",
        default=None,
        help="Node ID list for .npy embeddings (ignored for .npz).",
    )
    parser.add_argument("--kg-hop-expansion", type=int, default=0)
    parser.add_argument("--kg-expansion-max-nodes", type=int, default=None)
    parser.add_argument("--kg-expansion-verbose", action="store_true")
    parser.add_argument("--kg-workers", type=int, default=None)
    parser.add_argument(
        "--disease-token-position",
        choices=["first", "last", "none"],
        default=None,
        help="Optionally inject disease embedding as a token in the LSTM sequence.",
    )
    parser.add_argument(
        "--concat-disease-after-lstm",
        choices=["true", "false"],
        default=None,
        help="Whether to concat disease embedding after LSTM (default: true).",
    )
    parser.add_argument(
        "--kg-backend",
        choices=["auto", "pecanpy", "node2vec"],
        default="auto",
        help="Force KG embedding backend.",
    )
    parser.add_argument(
        "--kg-cache-path",
        default=os.path.join("artifacts", "kg_embeddings.npz"),
        help="Node2Vec cache file path to share with train.py.",
    )
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--per-drug-count-metrics",
        action="store_true",
        help="Report LSTM metrics grouped by drug count buckets.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of replicate train/holdout splits to run.",
    )
    parser.add_argument(
        "--replicate-seed-step",
        type=int,
        default=1,
        help="Seed increment per replicate (seed + step * i).",
    )
    parser.add_argument(
        "--single-therapy-indications",
        default=None,
        help="Optional RENCI single-therapy indications CSV.",
    )
    parser.add_argument(
        "--single-therapy-contraindications",
        default=None,
        help="Optional RENCI single-therapy contraindications CSV.",
    )
    parser.add_argument(
        "--twosides-contraindications",
        default="twosides_ddi_prefixed_normalized.csv",
        help="Optional TWOSIDES normalized contraindication-like interaction CSV.",
    )
    parser.add_argument(
        "--alias-index",
        default="artifacts/precomputed_embeddings/topological/equivalent_id_to_node_id.parquet",
        help="Alias-to-canonical node ID parquet used to resolve equivalent IDs everywhere.",
    )
    parser.add_argument(
        "--enable-mixed-negatives",
        action="store_true",
        help="Mix sourced negatives with randomized negatives.",
    )
    parser.add_argument(
        "--random-negative-ratio",
        type=float,
        default=1.0,
        help="Randomized negatives / sourced negatives ratio when mixed negatives are enabled.",
    )
    parser.add_argument(
        "--random-negative-strategy",
        choices=["disease_shuffle"],
        default="disease_shuffle",
        help="Strategy for randomized negative generation.",
    )
    parser.add_argument(
        "--save-mixed-dataset-details",
        action="store_true",
        help="Write mixed-negative source/report stats to output directory.",
    )
    parser.add_argument(
        "--extra-drug-ids-file",
        default=None,
        help=(
            "Optional newline-delimited file of extra drug IDs to include in the "
            "LSTM embedding/vocab tables when they exist in the precomputed embeddings."
        ),
    )
    parser.add_argument(
        "--extra-disease-ids-file",
        default=None,
        help=(
            "Optional newline-delimited file of extra disease IDs to include in the "
            "LSTM embedding/vocab tables when they exist in the precomputed embeddings."
        ),
    )
    return parser.parse_args()


def load_id_file(path: str | None) -> List[str]:
    if path is None:
        return []
    values: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                values.append(line)
    return sorted(set(values))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_idx_to_id(mapping: Dict[str, int], pad_token: str | None = None) -> List[str]:
    size = len(mapping) + (1 if pad_token else 0)
    idx_to_id = ["" for _ in range(size)]
    if pad_token is not None:
        idx_to_id[0] = pad_token
    for entity_id, idx in mapping.items():
        idx_to_id[idx] = entity_id
    return idx_to_id


def prepare_test_train_split(
    df: data_lib.pd.DataFrame, test_frac: float, seed: int, output_dir: str
) -> Dict[str, data_lib.pd.DataFrame]:
    two_drug_mask = df["drug_set"].apply(lambda ds: len(ds) == 2)
    two_drug_df = df[two_drug_mask]
    if two_drug_df.empty:
        raise ValueError("Filtered dataset contains no two-drug examples for test set.")
    test_size = max(1, int(len(two_drug_df) * test_frac))
    test_df = two_drug_df.sample(n=test_size, random_state=seed)
    np.save(
        os.path.join(output_dir, "two_drug_test_idx.npy"),
        test_df.index.to_numpy(),
    )
    train_df = df.drop(index=test_df.index)
    return {"train": train_df, "test": test_df}


def build_rf_features(
    df: data_lib.pd.DataFrame,
    drug_to_idx: Dict[str, int],
    disease_to_idx: Dict[str, int],
    drug_embeddings: np.ndarray,
    disease_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for row in df.itertuples(index=False):
        drug_set = list(row.drug_set)
        if len(drug_set) != 2:
            continue
        d1, d2 = drug_set
        emb = np.concatenate(
            (
                drug_embeddings[drug_to_idx[d1]],
                drug_embeddings[drug_to_idx[d2]],
                disease_embeddings[disease_to_idx[row.condition_id_norm]],
            )
        )
        rows.append(emb)
        labels.append(row.label)
    if not rows:
        raise ValueError("RF training set contains no 2-drug examples.")
    return np.stack(rows, axis=0), np.array(labels, dtype=np.int64)


def fit_random_forest(
    X: np.ndarray, y: np.ndarray, est: int, max_depth: int, seed: int
) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=est,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def build_pair_feature_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
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


def train_pair_mlp(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, object],
    input_dim: int,
    device: torch.device,
    output_dir: str,
) -> Tuple[model_lib.PairEmbeddingMLPClassifier, str]:
    model = model_lib.PairEmbeddingMLPClassifier(
        input_dim=input_dim,
        hidden_dim=int(config["pair_mlp_hidden_dim"]),
        num_layers=int(config["pair_mlp_layers"]),
        dropout=float(config["pair_mlp_dropout"]),
        init_sigma=config.get("pair_mlp_init_sigma"),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["pair_mlp_learning_rate"]),
        weight_decay=float(config["pair_mlp_weight_decay"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc = float("-inf")
    best_path = os.path.join(output_dir, "pair_mlp_best.pt")
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
        val_metrics = evaluate_pair_mlp(model, val_loader, device)
        print(
            f"PairMLP Epoch {epoch:02d} | loss={total_loss/len(train_loader):.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_sens={val_metrics['sensitivity']:.4f} | val_spec={val_metrics['specificity']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        if not np.isnan(val_metrics["roc_auc"]) and val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": int(config["pair_mlp_hidden_dim"]),
                    "num_layers": int(config["pair_mlp_layers"]),
                    "dropout": float(config["pair_mlp_dropout"]),
                    "init_sigma": config.get("pair_mlp_init_sigma"),
                    "learning_rate": float(config["pair_mlp_learning_rate"]),
                    "weight_decay": float(config["pair_mlp_weight_decay"]),
                    "epochs": int(config["pair_mlp_epochs"]),
                },
                best_path,
            )
    return model, best_path


def evaluate_model_with_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for drug_seq, lengths, disease_features, labels in loader:
            drug_seq = drug_seq.to(device)
            lengths = lengths.to(device)
            disease_features = disease_features.to(device)
            logits = model(drug_seq, lengths, disease_features)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    if not all_probs:
        return {"roc_auc": float("nan"), "accuracy": float("nan")}
    return utils.compute_metrics(np.concatenate(all_labels), np.concatenate(all_probs))


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    all_lengths = []
    with torch.no_grad():
        for drug_seq, lengths, disease_features, labels in loader:
            drug_seq = drug_seq.to(device)
            lengths = lengths.to(device)
            disease_features = disease_features.to(device)
            logits = model(drug_seq, lengths, disease_features)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_lengths.append(lengths.cpu().numpy())
    if not all_probs:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )
    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_lengths),
    )


def compute_metrics_safe(labels: np.ndarray, probs: np.ndarray) -> Dict[str, object]:
    from sklearn.metrics import roc_auc_score

    if len(labels) == 0:
        return {
            "roc_auc": float("nan"),
            "accuracy": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        }

    labels = labels.astype(np.int64)
    preds = (probs >= 0.5).astype(np.int64)
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tp = int(np.sum((labels == 1) & (preds == 1)))
    accuracy = float((tn + tp) / max(len(labels), 1))
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    denom = 2 * tp + fp + fn
    f1 = float(2 * tp / denom) if denom > 0 else float("nan")
    roc_auc = float("nan")
    if len(np.unique(labels)) == 2:
        roc_auc = float(roc_auc_score(labels, probs))
    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def compute_bucket_metrics(
    labels: np.ndarray, probs: np.ndarray, lengths: np.ndarray
) -> Dict[str, Dict[str, object]]:
    buckets = [
        ("1", lambda x: x == 1),
        ("2", lambda x: x == 2),
        ("3-4", lambda x: (x >= 3) & (x <= 4)),
        (">=5", lambda x: x >= 5),
    ]
    metrics_by_bucket: Dict[str, Dict[str, object]] = {}
    lengths = lengths.astype(np.int64)
    for name, predicate in buckets:
        mask = predicate(lengths)
        bucket_labels = labels[mask]
        bucket_probs = probs[mask]
        metrics = compute_metrics_safe(bucket_labels, bucket_probs)
        metrics["n"] = int(mask.sum())
        metrics_by_bucket[name] = metrics
    return metrics_by_bucket


class DirectEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        drug_sequences: List[np.ndarray],
        disease_embeddings: List[np.ndarray],
        labels: List[int],
    ) -> None:
        self.drug_sequences = drug_sequences
        self.disease_embeddings = disease_embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.drug_sequences[idx], dtype=torch.float32),
            torch.tensor(self.disease_embeddings[idx], dtype=torch.float32),
            torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        )


def collate_direct_embedding_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    drug_seqs, disease_embeddings, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in drug_seqs], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(
        drug_seqs, batch_first=True, padding_value=0.0
    )
    disease_tensor = torch.stack(disease_embeddings)
    label_tensor = torch.stack(labels)
    return padded, lengths, disease_tensor, label_tensor


def embed_examples_direct(
    examples: List[data_lib.LabeledExample],
    node_to_idx: Dict[str, int],
    node_vectors: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    drug_sequences: List[np.ndarray] = []
    disease_embeddings: List[np.ndarray] = []
    labels: List[int] = []
    for example in examples:
        drug_sequences.append(
            np.stack(
                [np.asarray(node_vectors[node_to_idx[drug_id]], dtype=np.float32) for drug_id in example.drug_ids],
                axis=0,
            )
        )
        disease_embeddings.append(
            np.asarray(node_vectors[node_to_idx[example.disease_id]], dtype=np.float32)
        )
        labels.append(example.label)
    return drug_sequences, disease_embeddings, labels


def train_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    drug_embedding_dim: int,
    disease_embedding_dim: int,
    device: torch.device,
    output_dir: str,
) -> Tuple[torch.nn.Module, str]:
    model = model_lib.PolypharmacyDirectEmbeddingLSTMClassifier(
        drug_embedding_dim=drug_embedding_dim,
        disease_embedding_dim=disease_embedding_dim,
        lstm_hidden_dim=config["lstm_hidden_dim"],
        mlp_hidden_dim=config["mlp_hidden_dim"],
        mlp_layers=config["mlp_layers"],
        dropout=config["dropout"],
        disease_token_position=config.get("disease_token_position"),
        concat_disease_after_lstm=config.get("concat_disease_after_lstm", True),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc = float("-inf")
    best_path = os.path.join(output_dir, "best_model.pt")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for drug_seq, lengths, disease_idx, labels in train_loader:
            drug_seq = drug_seq.to(device)
            lengths = lengths.to(device)
            disease_idx = disease_idx.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(drug_seq, lengths, disease_idx)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_metrics = evaluate_model_with_loader(model, val_loader, device)
        print(
            f"Epoch {epoch:02d} | loss={total_loss/len(train_loader):.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_sens={val_metrics['sensitivity']:.4f} | val_spec={val_metrics['specificity']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        if not np.isnan(val_metrics["roc_auc"]) and val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "uses_direct_embeddings": True,
                    "drug_embedding_dim": int(drug_embedding_dim),
                    "disease_embedding_dim": int(disease_embedding_dim),
                    "lstm_hidden_dim": config["lstm_hidden_dim"],
                    "mlp_hidden_dim": config["mlp_hidden_dim"],
                    "mlp_layers": config["mlp_layers"],
                    "dropout": config["dropout"],
                    "disease_token_position": config.get("disease_token_position"),
                    "concat_disease_after_lstm": config.get(
                        "concat_disease_after_lstm", True
                    ),
                },
                best_path,
            )
    return model, best_path


def aggregate_metrics(metrics_list: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    metric_keys = ["roc_auc", "accuracy", "sensitivity", "specificity", "f1"]
    for key in metric_keys:
        values = np.array([float(m[key]) for m in metrics_list], dtype=float)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "n": 0}
            continue
        mean = float(np.mean(valid))
        if len(valid) > 1:
            std = float(np.std(valid, ddof=1))
            sem = float(std / np.sqrt(len(valid)))
        else:
            std = 0.0
            sem = 0.0
        summary[key] = {"mean": mean, "std": std, "sem": sem, "n": int(len(valid))}
    return summary


def format_mean_sem(value: Dict[str, float]) -> str:
    return f"{value['mean']:.4f}±{value['sem']:.4f}"


def print_summary(label: str, summary: Dict[str, Dict[str, float]]) -> None:
    print(f"{label} summary (mean±SEM):")
    print(
        "  "
        + " | ".join(
            [
                f"auc={format_mean_sem(summary['roc_auc'])}",
                f"acc={format_mean_sem(summary['accuracy'])}",
                f"sens={format_mean_sem(summary['sensitivity'])}",
                f"spec={format_mean_sem(summary['specificity'])}",
                f"f1={format_mean_sem(summary['f1'])}",
            ]
        )
    )


def main() -> None:
    args = parse_args()
    config = config_lib.load_config(args.config)
    if args.disease_token_position is not None:
        config["disease_token_position"] = (
            None if args.disease_token_position == "none" else args.disease_token_position
        )
    if args.concat_disease_after_lstm is not None:
        config["concat_disease_after_lstm"] = args.concat_disease_after_lstm == "true"
    print("Resolved experiment config:\n" + json.dumps(config, indent=2))
    utils.set_seeds(args.seed)
    ensure_dir(args.output_dir)
    utils.save_json(os.path.join(args.output_dir, "config.json"), config)

    dedupe_report: Dict[str, object] = {}
    deduped_df, conflict_count = data_lib.load_deduped_dataframe(
        args.indications,
        args.contraindications,
        single_therapy_indications_path=args.single_therapy_indications,
        single_therapy_contraindications_path=args.single_therapy_contraindications,
        twosides_contraindications_path=args.twosides_contraindications,
        alias_index_path=args.alias_index,
        enable_mixed_negatives=args.enable_mixed_negatives,
        random_negative_ratio=args.random_negative_ratio,
        random_negative_strategy=args.random_negative_strategy,
        seed=args.seed,
        report_out=dedupe_report,
    )
    print(f"Conflict resolution: {conflict_count}")
    if args.enable_mixed_negatives:
        print(
            "Mixed negatives enabled: "
            f"strategy={args.random_negative_strategy}, ratio={args.random_negative_ratio}"
        )
    if args.save_mixed_dataset_details:
        utils.save_json(
            os.path.join(args.output_dir, "mixed_negative_report.json"),
            dedupe_report,
        )

    required_drugs = set(
        itertools.chain.from_iterable(deduped_df["drug_set"])  # type: ignore[arg-type]
    )
    required_diseases = set(deduped_df["condition_id_norm"])
    required_nodes = required_drugs.union(required_diseases)

    node_ids = None
    node_vectors = None
    pruned_edges = None
    if args.kg_embeddings:
        node_ids, node_vectors = kg_lib.load_precomputed_embeddings(
            args.kg_embeddings, args.kg_embedding_ids
        )
        kg_nodes = set(node_ids)
        print(
            "KG coverage filtering: using precomputed embedding node IDs "
            f"(nodes={len(kg_nodes)})"
        )
    else:
        edges = kg_lib.load_edges(args.kg, src_col=None, dst_col=None)
        initial_edges = len(edges)
        expanded_nodes = required_nodes
        if args.kg_hop_expansion > 0:
            expanded_nodes, hop_logs, truncated = kg_lib.expand_node_set(
                edges,
                required_nodes,
                hops=args.kg_hop_expansion,
                max_nodes=args.kg_expansion_max_nodes,
            )
            print(
                f"KG hop expansion: k={args.kg_hop_expansion}, nodes required={len(required_nodes)} -> expanded={len(expanded_nodes)}"
            )
            if args.kg_expansion_verbose:
                for hop, added, cum in hop_logs:
                    print(f"hop {hop}: +{added} nodes (cum {cum})")
                if truncated:
                    print("KG expansion stopped early due to max node cap.")

        pruned_edges = kg_lib.prune_edges_to_nodes(edges, expanded_nodes)
        if len(pruned_edges) < initial_edges:
            print(
                f"KG pruning: reduced edges from {initial_edges} to {len(pruned_edges)} "
                f"covering {len(expanded_nodes)} nodes"
            )

        kg_cache_dir = os.path.dirname(args.kg_cache_path)
        if kg_cache_dir:
            os.makedirs(kg_cache_dir, exist_ok=True)

        node_ids, node_vectors = kg_lib.load_or_build_kg_embeddings(
            args.kg,
            cache_path=args.kg_cache_path,
            embedding_dim=config["embedding_dim"],
            walk_length=config["kg_walk_length"],
            num_walks=config["kg_num_walks"],
            p=config["kg_p"],
            q=config["kg_q"],
            context_window=config["kg_context_window"],
            min_count=config["kg_min_count"],
            workers=args.kg_workers or config["kg_workers"],
            seed=args.seed,
            src_col=None,
            dst_col=None,
            edges=pruned_edges,
            backend=args.kg_backend,
        )

        kg_nodes = kg_lib.extract_kg_nodes(pruned_edges)
    filtered_df, dropped_df, drop_stats = data_lib.filter_by_kg_coverage(
        deduped_df, kg_nodes
    )
    ensure_dir(args.output_dir)
    filtered_df.to_csv(os.path.join(args.output_dir, "filtered_dataset.csv"), index=False)
    dropped_df.to_csv(os.path.join(args.output_dir, "dropped_rows.csv"), index=False)
    print(
        "KG coverage filtering: "
        f"dropped={drop_stats['num_dropped']} "
        f"({drop_stats['percent_dropped']:.2%})"
    )
    if drop_stats["missing_prefixes"]:
        print(f"Most common missing prefixes: {drop_stats['missing_prefixes']}")

    filtered_examples = data_lib.dataframe_to_examples(filtered_df)
    requested_extra_drug_ids = load_id_file(args.extra_drug_ids_file)
    requested_extra_disease_ids = load_id_file(args.extra_disease_ids_file)
    extra_drug_ids = [drug_id for drug_id in requested_extra_drug_ids if drug_id in kg_nodes]
    extra_disease_ids = [
        disease_id for disease_id in requested_extra_disease_ids if disease_id in kg_nodes
    ]
    skipped_extra_drug_ids = sorted(set(requested_extra_drug_ids) - set(extra_drug_ids))
    skipped_extra_disease_ids = sorted(
        set(requested_extra_disease_ids) - set(extra_disease_ids)
    )
    if extra_drug_ids:
        print(
            "Including extra drug IDs in embedding vocab: "
            f"{len(extra_drug_ids)} kept from {len(requested_extra_drug_ids)} requested"
        )
    if extra_disease_ids:
        print(
            "Including extra disease IDs in embedding vocab: "
            f"{len(extra_disease_ids)} kept from {len(requested_extra_disease_ids)} requested"
        )
    if skipped_extra_drug_ids:
        print(f"Skipped extra drug IDs missing from KG embeddings: {skipped_extra_drug_ids}")
    if skipped_extra_disease_ids:
        print(
            "Skipped extra disease IDs missing from KG embeddings: "
            f"{skipped_extra_disease_ids}"
        )
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    embedding_dim = node_vectors.shape[1]
    rng = np.random.RandomState(args.seed)
    drug_to_idx, disease_to_idx = data_lib.build_mappings(
        filtered_examples,
        extra_drug_ids=extra_drug_ids,
        extra_disease_ids=extra_disease_ids,
    )
    drug_idx_to_id = build_idx_to_id(drug_to_idx, pad_token="<PAD>")
    disease_idx_to_id = build_idx_to_id(disease_to_idx)
    drug_embeddings = kg_lib.build_entity_embedding(
        entity_ids=drug_idx_to_id,
        node_to_idx=node_to_idx,
        node_embeddings=node_vectors,
        embedding_dim=embedding_dim,
        rng=rng,
        pad_idx=0,
    )
    disease_embeddings = kg_lib.build_entity_embedding(
        entity_ids=disease_idx_to_id,
        node_to_idx=node_to_idx,
        node_embeddings=node_vectors,
        embedding_dim=embedding_dim,
        rng=rng,
        pad_idx=None,
    )

    if embedding_dim != config["embedding_dim"]:
        config["embedding_dim"] = embedding_dim

    utils.save_json(
        os.path.join(args.output_dir, "drug_vocab.json"),
        {"ids": drug_idx_to_id},
    )
    utils.save_json(
        os.path.join(args.output_dir, "disease_vocab.json"),
        {"ids": disease_idx_to_id},
    )
    np.save(os.path.join(args.output_dir, "drug_embeddings.npy"), drug_embeddings)
    np.save(os.path.join(args.output_dir, "disease_embeddings.npy"), disease_embeddings)
    utils.save_json(
        os.path.join(args.output_dir, "extra_vocab_metadata.json"),
        {
            "requested_extra_drug_ids": requested_extra_drug_ids,
            "requested_extra_disease_ids": requested_extra_disease_ids,
            "included_extra_drug_ids": extra_drug_ids,
            "included_extra_disease_ids": extra_disease_ids,
            "skipped_extra_drug_ids": skipped_extra_drug_ids,
            "skipped_extra_disease_ids": skipped_extra_disease_ids,
        },
    )

    def encode_split(df: data_lib.pd.DataFrame) -> Tuple[List[List[int]], List[int], List[int]]:
        examples = data_lib.dataframe_to_examples(df)
        return data_lib.encode_examples(examples, drug_to_idx, disease_to_idx)

    def embed_split_direct(
        df: data_lib.pd.DataFrame,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        examples = data_lib.dataframe_to_examples(df)
        return embed_examples_direct(examples, node_to_idx, node_vectors)

    filtered_positions = {idx: pos for pos, idx in enumerate(filtered_df.index)}
    filtered_df.to_csv(os.path.join(args.output_dir, "filtered_dataset_run.csv"), index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rf_metrics_all: List[Dict[str, object]] = []
    pair_mlp_metrics_all: List[Dict[str, object]] = []
    lstm_metrics_all: List[Dict[str, object]] = []
    bucket_metrics_val: Dict[str, List[Dict[str, object]]] = {
        "1": [],
        "2": [],
        "3-4": [],
        ">=5": [],
    }
    bucket_metrics_test: Dict[str, List[Dict[str, object]]] = {
        "1": [],
        "2": [],
        "3-4": [],
        ">=5": [],
    }
    replicate_payloads: List[Dict[str, object]] = []

    for rep in range(args.replicates):
        rep_seed = args.seed + rep * args.replicate_seed_step
        run_output_dir = (
            args.output_dir
            if args.replicates == 1
            else os.path.join(args.output_dir, f"run_{rep:02d}")
        )
        ensure_dir(run_output_dir)
        utils.set_seeds(rep_seed)
        print(f"\n=== Replicate {rep + 1}/{args.replicates} (seed={rep_seed}) ===")

        split_data = prepare_test_train_split(
            filtered_df, args.test_frac, rep_seed, run_output_dir
        )
        train_df = split_data["train"]
        test_df = split_data["test"]

        stratify_col = train_df["label"] if train_df["label"].nunique() > 1 else None
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.1,
            random_state=rep_seed,
            stratify=stratify_col,
        )

        train_seqs, train_diseases, train_labels = encode_split(train_df)
        val_seqs, val_diseases, val_labels = encode_split(val_df)
        test_seqs, test_diseases, test_labels = encode_split(test_df)
        train_seq_embs, train_dis_embs, train_lstm_labels = embed_split_direct(train_df)
        val_seq_embs, val_dis_embs, val_lstm_labels = embed_split_direct(val_df)
        test_seq_embs, test_dis_embs, test_lstm_labels = embed_split_direct(test_df)

        train_idx = np.array([filtered_positions[idx] for idx in train_df.index], dtype=np.int64)
        val_idx = np.array([filtered_positions[idx] for idx in val_df.index], dtype=np.int64)
        test_idx = np.array([filtered_positions[idx] for idx in test_df.index], dtype=np.int64)
        np.savez_compressed(
            os.path.join(run_output_dir, "splits.npz"),
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            num_examples=len(filtered_df),
        )

        if not train_seqs or not val_seqs or not test_seqs:
            raise ValueError("One of the splits became empty after encoding.")

        train_dataset = DirectEmbeddingDataset(
            train_seq_embs, train_dis_embs, train_lstm_labels
        )
        val_dataset = DirectEmbeddingDataset(
            val_seq_embs, val_dis_embs, val_lstm_labels
        )
        test_dataset = DirectEmbeddingDataset(
            test_seq_embs, test_dis_embs, test_lstm_labels
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_direct_embedding_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_direct_embedding_batch,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_direct_embedding_batch,
        )

        print(
            "Dataset split sizes (train/val/test): "
            f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}"
        )

        rf_train_df = train_df[train_df["drug_set"].apply(lambda ds: len(ds) == 2)]
        print(
            f"RF train candidates: {len(rf_train_df)} two-drug examples, "
            f"test set: {len(test_df)} two-drug examples"
        )
        X_rf_train, y_rf_train = build_rf_features(
            rf_train_df, drug_to_idx, disease_to_idx, drug_embeddings, disease_embeddings
        )
        rf_val_df = val_df[val_df["drug_set"].apply(lambda ds: len(ds) == 2)]
        if rf_val_df.empty:
            raise ValueError("Validation split contains no 2-drug examples for pairwise baselines.")
        X_rf_val, y_rf_val = build_rf_features(
            rf_val_df, drug_to_idx, disease_to_idx, drug_embeddings, disease_embeddings
        )
        rf_model = fit_random_forest(
            X_rf_train,
            y_rf_train,
            est=args.rf_estimators,
            max_depth=args.rf_max_depth,
            seed=rep_seed,
        )
        rf_model_path = os.path.join(run_output_dir, "rf_model.pkl")
        with open(rf_model_path, "wb") as handle:
            pickle.dump(rf_model, handle)
        rf_meta = {
            "seed": rep_seed,
            "n_estimators": args.rf_estimators,
            "max_depth": args.rf_max_depth,
            "train_examples": int(len(y_rf_train)),
            "feature_dim": int(X_rf_train.shape[1]),
            "mixed_negatives_enabled": bool(args.enable_mixed_negatives),
            "random_negative_ratio": float(args.random_negative_ratio),
            "random_negative_strategy": str(args.random_negative_strategy),
        }
        utils.save_json(os.path.join(run_output_dir, "rf_model_metadata.json"), rf_meta)
        print(f"Saved RF model to {rf_model_path}")
        X_rf_test, y_rf_test = build_rf_features(
            test_df, drug_to_idx, disease_to_idx, drug_embeddings, disease_embeddings
        )
        rf_probs = rf_model.predict_proba(X_rf_test)[:, 1]
        rf_metrics = utils.compute_metrics(y_rf_test, rf_probs)
        rf_metrics_all.append(rf_metrics)
        print(
            f"RF on held-out 2-drug test | auc={rf_metrics['roc_auc']:.4f} | "
            f"acc={rf_metrics['accuracy']:.4f} | sens={rf_metrics['sensitivity']:.4f} | "
            f"spec={rf_metrics['specificity']:.4f} | f1={rf_metrics['f1']:.4f} | "
            f"confusion={rf_metrics['confusion']}"
        )

        pair_train_loader = build_pair_feature_loader(
            X_rf_train,
            y_rf_train,
            batch_size=int(config["pair_mlp_batch_size"]),
            shuffle=True,
        )
        pair_val_loader = build_pair_feature_loader(
            X_rf_val,
            y_rf_val,
            batch_size=int(config["pair_mlp_batch_size"]),
            shuffle=False,
        )
        pair_test_loader = build_pair_feature_loader(
            X_rf_test,
            y_rf_test,
            batch_size=int(config["pair_mlp_batch_size"]),
            shuffle=False,
        )
        pair_mlp_model, pair_mlp_best_path = train_pair_mlp(
            pair_train_loader,
            pair_val_loader,
            config,
            input_dim=int(X_rf_train.shape[1]),
            device=device,
            output_dir=run_output_dir,
        )
        pair_mlp_metrics = evaluate_pair_mlp(pair_mlp_model, pair_test_loader, device)
        pair_mlp_metrics_all.append(pair_mlp_metrics)
        utils.save_json(
            os.path.join(run_output_dir, "pair_mlp_metadata.json"),
            {
                "seed": rep_seed,
                "train_examples": int(len(y_rf_train)),
                "val_examples": int(len(y_rf_val)),
                "test_examples": int(len(y_rf_test)),
                "feature_dim": int(X_rf_train.shape[1]),
                "hidden_dim": int(config["pair_mlp_hidden_dim"]),
                "num_layers": int(config["pair_mlp_layers"]),
                "dropout": float(config["pair_mlp_dropout"]),
                "epochs": int(config["pair_mlp_epochs"]),
                "batch_size": int(config["pair_mlp_batch_size"]),
                "learning_rate": float(config["pair_mlp_learning_rate"]),
                "weight_decay": float(config["pair_mlp_weight_decay"]),
                "init_sigma": config.get("pair_mlp_init_sigma"),
                "mixed_negatives_enabled": bool(args.enable_mixed_negatives),
                "random_negative_ratio": float(args.random_negative_ratio),
                "random_negative_strategy": str(args.random_negative_strategy),
            },
        )
        print(
            f"PairMLP on held-out 2-drug test | auc={pair_mlp_metrics['roc_auc']:.4f} | "
            f"acc={pair_mlp_metrics['accuracy']:.4f} | sens={pair_mlp_metrics['sensitivity']:.4f} | "
            f"spec={pair_mlp_metrics['specificity']:.4f} | f1={pair_mlp_metrics['f1']:.4f} | "
            f"confusion={pair_mlp_metrics['confusion']}"
        )
        print(f"Best PairMLP checkpoint saved to {pair_mlp_best_path}")

        lstm_model, lstm_best_path = train_lstm(
            train_loader,
            val_loader,
            config,
            drug_embedding_dim=int(embedding_dim),
            disease_embedding_dim=int(embedding_dim),
            device=device,
            output_dir=run_output_dir,
        )
        test_probs, test_labels, test_lengths = collect_predictions(
            lstm_model, test_loader, device
        )
        lstm_metrics = utils.compute_metrics(test_labels, test_probs)
        lstm_metrics_all.append(lstm_metrics)
        print(
            f"LSTM on held-out 2-drug test | auc={lstm_metrics['roc_auc']:.4f} | "
            f"acc={lstm_metrics['accuracy']:.4f} | sens={lstm_metrics['sensitivity']:.4f} | "
            f"spec={lstm_metrics['specificity']:.4f} | f1={lstm_metrics['f1']:.4f} | "
            f"confusion={lstm_metrics['confusion']}"
        )
        print(f"Best LSTM checkpoint saved to {lstm_best_path}")

        if args.per_drug_count_metrics:
            val_probs, val_labels, val_lengths = collect_predictions(
                lstm_model, val_loader, device
            )
            val_bucket_metrics = compute_bucket_metrics(
                val_labels, val_probs, val_lengths
            )
            test_bucket_metrics = compute_bucket_metrics(
                test_labels, test_probs, test_lengths
            )
            print("LSTM per-drug-count metrics (validation):")
            for bucket, metrics in val_bucket_metrics.items():
                print(
                    f"  drugs={bucket:<3} n={metrics['n']:<4} | "
                    f"auc={metrics['roc_auc']:.4f} | acc={metrics['accuracy']:.4f} | "
                    f"sens={metrics['sensitivity']:.4f} | spec={metrics['specificity']:.4f} | "
                    f"f1={metrics['f1']:.4f}"
                )
            print("LSTM per-drug-count metrics (test):")
            for bucket, metrics in test_bucket_metrics.items():
                print(
                    f"  drugs={bucket:<3} n={metrics['n']:<4} | "
                    f"auc={metrics['roc_auc']:.4f} | acc={metrics['accuracy']:.4f} | "
                    f"sens={metrics['sensitivity']:.4f} | spec={metrics['specificity']:.4f} | "
                    f"f1={metrics['f1']:.4f}"
                )
            for bucket in bucket_metrics_val:
                bucket_metrics_val[bucket].append(val_bucket_metrics[bucket])
                bucket_metrics_test[bucket].append(test_bucket_metrics[bucket])

        replicate_payloads.append(
            {
                "replicate": rep,
                "seed": rep_seed,
                "run_dir": run_output_dir,
                "rf_metrics": rf_metrics,
                "pair_mlp_metrics": pair_mlp_metrics,
                "lstm_metrics": lstm_metrics,
                "lstm_val_bucket_metrics": val_bucket_metrics
                if args.per_drug_count_metrics
                else None,
                "lstm_test_bucket_metrics": test_bucket_metrics
                if args.per_drug_count_metrics
                else None,
                "test_size": len(test_df),
                "train_size": len(train_df),
                "val_size": len(val_df),
            }
        )
        utils.save_json(
            os.path.join(run_output_dir, "metrics.json"),
            {
                "replicate": rep,
                "seed": rep_seed,
                "rf_metrics": rf_metrics,
                "pair_mlp_metrics": pair_mlp_metrics,
                "lstm_metrics": lstm_metrics,
            },
        )

    if args.replicates > 1:
        rf_summary = aggregate_metrics(rf_metrics_all)
        pair_mlp_summary = aggregate_metrics(pair_mlp_metrics_all)
        lstm_summary = aggregate_metrics(lstm_metrics_all)
        print_summary("RF", rf_summary)
        print_summary("PairMLP", pair_mlp_summary)
        print_summary("LSTM", lstm_summary)
        if args.per_drug_count_metrics:
            print("LSTM per-drug-count summary (mean±SEM):")
            for bucket in bucket_metrics_val:
                val_summary = aggregate_metrics(bucket_metrics_val[bucket])
                test_summary = aggregate_metrics(bucket_metrics_test[bucket])
                val_n = float(np.mean([m["n"] for m in bucket_metrics_val[bucket]]))
                test_n = float(np.mean([m["n"] for m in bucket_metrics_test[bucket]]))
                print(
                    f"  drugs={bucket:<3} n_mean={val_n:.1f} | val "
                    f"auc={format_mean_sem(val_summary['roc_auc'])} | "
                    f"acc={format_mean_sem(val_summary['accuracy'])} | "
                    f"sens={format_mean_sem(val_summary['sensitivity'])} | "
                    f"spec={format_mean_sem(val_summary['specificity'])} | "
                    f"f1={format_mean_sem(val_summary['f1'])}"
                )
                print(
                    f"  drugs={bucket:<3} n_mean={test_n:.1f} | test "
                    f"auc={format_mean_sem(test_summary['roc_auc'])} | "
                    f"acc={format_mean_sem(test_summary['accuracy'])} | "
                    f"sens={format_mean_sem(test_summary['sensitivity'])} | "
                    f"spec={format_mean_sem(test_summary['specificity'])} | "
                    f"f1={format_mean_sem(test_summary['f1'])}"
                )
        utils.save_json(
            os.path.join(args.output_dir, "bag_summary.json"),
            {
                "replicates": args.replicates,
                "seed": args.seed,
                "replicate_seed_step": args.replicate_seed_step,
                "rf_summary": rf_summary,
                "pair_mlp_summary": pair_mlp_summary,
                "lstm_summary": lstm_summary,
                "lstm_bucket_summary": {
                    "validation": {
                        bucket: aggregate_metrics(bucket_metrics_val[bucket])
                        for bucket in bucket_metrics_val
                    },
                    "test": {
                        bucket: aggregate_metrics(bucket_metrics_test[bucket])
                        for bucket in bucket_metrics_test
                    },
                }
                if args.per_drug_count_metrics
                else None,
                "replicate_metrics": replicate_payloads,
            },
        )


if __name__ == "__main__":
    main()
