import argparse
import itertools
import json
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from polypharmacy import config as config_lib
from polypharmacy import data as data_lib
from polypharmacy import kg as kg_lib
from polypharmacy import model as model_lib
from polypharmacy import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on all drug counts except a held-out length."
    )
    parser.add_argument("--indications", default="indications_norm.csv")
    parser.add_argument("--contraindications", default="contraindications_norm.csv")
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
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--config", default=None, help="Optional JSON config override.")
    parser.add_argument("--edge-src-col", default=None)
    parser.add_argument("--edge-dst-col", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--kg-num-walks", type=int, default=None)
    parser.add_argument("--kg-walk-length", type=int, default=None)
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
        help="Node2Vec cache file path to share with train.py/experiment.py.",
    )
    parser.add_argument(
        "--kg-hop-expansion",
        type=int,
        default=0,
        help="Expand the KG node set by this many hops before pruning.",
    )
    parser.add_argument(
        "--kg-expansion-max-nodes",
        type=int,
        default=None,
        help="Stop expanding if more than this many nodes would be added.",
    )
    parser.add_argument(
        "--kg-expansion-verbose",
        action="store_true",
        help="Log node counts per hop during KG expansion.",
    )
    parser.add_argument(
        "--holdout-drug-count",
        required=True,
        help="Hold out a drug-count bucket (e.g., '1', '2', '3-4', '>=5').",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick smoke test with reduced data/compute.",
    )
    return parser.parse_args()


def build_index_to_id(mapping: Dict[str, int], pad_token: Optional[str] = None) -> list:
    size = len(mapping) + (1 if pad_token is not None else 0)
    idx_to_id = ["" for _ in range(size)]
    if pad_token is not None:
        idx_to_id[0] = pad_token
    for entity_id, idx in mapping.items():
        idx_to_id[idx] = entity_id
    return idx_to_id


def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for drug_seq, lengths, disease_idx, batch_labels in loader:
            drug_seq = drug_seq.to(device)
            lengths = lengths.to(device)
            disease_idx = disease_idx.to(device)
            logits = model(drug_seq, lengths, disease_idx)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch_labels.numpy())
    if not all_probs:
        return {"roc_auc": float("nan"), "accuracy": float("nan")}
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return compute_metrics_safe(labels, probs)


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


def parse_holdout_spec(spec: str) -> Tuple[Callable[[int], bool], str]:
    cleaned = spec.strip()
    if cleaned.startswith(">="):
        value = int(cleaned[2:].strip())
        return (lambda x: x >= value), f">={value}"
    if "-" in cleaned:
        left, right = cleaned.split("-", 1)
        lo = int(left.strip())
        hi = int(right.strip())
        return (lambda x: (x >= lo) and (x <= hi)), f"{lo}-{hi}"
    value = int(cleaned)
    return (lambda x: x == value), str(value)


def deterministic_train_val_split(
    num_examples: int, seed: int, train_frac: float, val_frac: float
) -> Tuple[np.ndarray, np.ndarray]:
    if num_examples <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    total = train_frac + val_frac
    if total <= 0:
        raise ValueError("train_frac + val_frac must be > 0")
    train_frac = train_frac / total
    rng = np.random.RandomState(seed)
    indices = np.arange(num_examples)
    rng.shuffle(indices)
    if num_examples < 2:
        return indices[:1], indices[1:]
    train_end = max(1, int(num_examples * train_frac))
    train_idx = indices[:train_end]
    val_idx = indices[train_end:]
    return train_idx, val_idx


def main() -> None:
    args = parse_args()
    config = config_lib.load_config(args.config)

    if args.quick:
        if args.max_examples is None:
            args.max_examples = 500
        if args.max_edges is None:
            args.max_edges = 200000
        if args.epochs is None:
            args.epochs = 1
        if args.batch_size is None:
            args.batch_size = 32
        if args.kg_num_walks is None:
            args.kg_num_walks = 5
        if args.kg_walk_length is None:
            args.kg_walk_length = 10
        if args.kg_workers is None:
            args.kg_workers = 2

    override_map = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "kg_num_walks": args.kg_num_walks,
        "kg_walk_length": args.kg_walk_length,
        "kg_workers": args.kg_workers,
    }
    if args.disease_token_position is not None:
        config["disease_token_position"] = (
            None if args.disease_token_position == "none" else args.disease_token_position
        )
    if args.concat_disease_after_lstm is not None:
        config["concat_disease_after_lstm"] = args.concat_disease_after_lstm == "true"
    for key, value in override_map.items():
        if value is not None:
            config[key] = value
    print("Resolved generalize config:\n" + json.dumps(config, indent=2))

    utils.set_seeds(config["seed"])
    utils.ensure_dir(args.output_dir)

    deduped_df, conflict_count = data_lib.load_deduped_dataframe(
        args.indications,
        args.contraindications,
        single_therapy_indications_path=args.single_therapy_indications,
        single_therapy_contraindications_path=args.single_therapy_contraindications,
    )
    print(f"Conflict resolution: {conflict_count} conflicting keys set to label=0")
    deduped_path = os.path.join(args.output_dir, "deduped_dataset.csv")
    deduped_df.to_csv(deduped_path, index=False)

    required_drugs = set(
        itertools.chain.from_iterable(deduped_df["drug_set"])  # type: ignore[arg-type]
    )
    required_diseases = set(deduped_df["condition_id_norm"])
    required_nodes = required_drugs.union(required_diseases)

    node_ids = None
    node_vectors = None
    edges = None
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
        edges = kg_lib.load_edges(
            args.kg, src_col=args.edge_src_col, dst_col=args.edge_dst_col
        )
        initial_edge_count = len(edges)
        expanded_nodes = required_nodes
        hop_logs = []
        truncated = False
        if args.kg_hop_expansion > 0:
            expanded_nodes, hop_logs, truncated = kg_lib.expand_node_set(
                edges,
                required_nodes,
                hops=args.kg_hop_expansion,
                max_nodes=args.kg_expansion_max_nodes,
            )
            print(
                "KG hop expansion: "
                f"k={args.kg_hop_expansion}, nodes: required={len(required_nodes)} -> expanded={len(expanded_nodes)}"
            )
            if args.kg_expansion_verbose:
                for hop, new_nodes, cum_nodes in hop_logs:
                    print(f"hop {hop}: +{new_nodes} nodes (cum {cum_nodes})")
            if truncated and args.kg_expansion_verbose:
                print(
                    "KG hop expansion stopped early because node count exceeded "
                    f"{args.kg_expansion_max_nodes}"
                )

        edges = kg_lib.prune_edges_to_nodes(edges, expanded_nodes)
        if len(edges) < initial_edge_count:
            print(
                f"KG node filtering: reduced edges from {initial_edge_count} to {len(edges)} "
                f"to cover {len(required_nodes)} dataset nodes"
            )
        if args.max_edges is not None and len(edges) > args.max_edges:
            edges = edges.sample(n=args.max_edges, random_state=config["seed"]).reset_index(
                drop=True
            )
            print(f"KG edge sampling: using {len(edges)} edges")
        kg_nodes = kg_lib.extract_kg_nodes(edges)

    filtered_df, dropped_df, drop_stats = data_lib.filter_by_kg_coverage(
        deduped_df, kg_nodes
    )
    dropped_path = os.path.join(args.output_dir, "dropped_rows.csv")
    dropped_df.to_csv(dropped_path, index=False)
    filtered_path = os.path.join(args.output_dir, "filtered_dataset.csv")
    filtered_df.to_csv(filtered_path, index=False)
    print(
        "KG coverage filtering: "
        f"dropped={drop_stats['num_dropped']} "
        f"({drop_stats['percent_dropped']:.2%})"
    )
    if drop_stats["missing_prefixes"]:
        print(f"Most common missing prefixes: {drop_stats['missing_prefixes']}")

    holdout_fn, holdout_label = parse_holdout_spec(args.holdout_drug_count)
    holdout_mask = filtered_df["drug_set"].apply(lambda ds: holdout_fn(len(ds)))
    holdout_bucket_df = filtered_df[holdout_mask].reset_index(drop=True)
    non_holdout_df = filtered_df[~holdout_mask].reset_index(drop=True)

    if len(holdout_bucket_df) == 0:
        raise ValueError(
            f"No examples found for holdout bucket '{holdout_label}'."
        )
    if len(non_holdout_df) == 0:
        raise ValueError(
            f"No non-holdout examples remain after filtering '{holdout_label}'."
        )

    total_examples = len(filtered_df)
    desired_test_size = max(1, int(total_examples * config["test_frac"]))
    if len(holdout_bucket_df) <= desired_test_size:
        test_df = holdout_bucket_df
        train_val_pool_df = non_holdout_df
        print(
            f"Holdout bucket size {len(holdout_bucket_df)} < desired test size "
            f"{desired_test_size}; using all holdout examples for test."
        )
    else:
        test_df = holdout_bucket_df.sample(
            n=desired_test_size, random_state=config["seed"]
        )
        remaining_holdout = holdout_bucket_df.drop(test_df.index)
        test_df = test_df.reset_index(drop=True)
        remaining_holdout = remaining_holdout.reset_index(drop=True)
        train_val_pool_df = pd.concat([non_holdout_df, remaining_holdout], ignore_index=True)
        print(
            f"Holdout bucket size {len(holdout_bucket_df)} >= desired test size "
            f"{desired_test_size}; using {len(test_df)} holdout examples for test "
            "and returning the rest to train/val."
        )

    if args.max_examples is not None and len(train_val_pool_df) > args.max_examples:
        train_val_pool_df = train_val_pool_df.sample(
            n=args.max_examples, random_state=config["seed"]
        ).reset_index(drop=True)
        print(
            f"Example sampling (train/val pool): using {len(train_val_pool_df)} examples"
        )

    train_idx, val_idx = deterministic_train_val_split(
        num_examples=len(train_val_pool_df),
        seed=config["seed"],
        train_frac=config["train_frac"],
        val_frac=config["val_frac"],
    )

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            "Not enough training examples after holdout to populate train/val splits."
        )

    train_df = train_val_pool_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_pool_df.iloc[val_idx].reset_index(drop=True)

    generalize_path = os.path.join(args.output_dir, "generalize_dataset.csv")
    generalize_df = pd.concat([train_val_pool_df, test_df], ignore_index=True)
    generalize_df.to_csv(generalize_path, index=False)
    train_df.to_csv(os.path.join(args.output_dir, "generalize_train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "generalize_val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "generalize_test.csv"), index=False)
    np.savez_compressed(
        os.path.join(args.output_dir, "generalize_splits.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=np.arange(len(test_df)) + len(train_val_pool_df),
        num_train_val=len(train_val_pool_df),
        num_test=len(test_df),
        holdout_spec=holdout_label,
    )
    utils.save_json(
        os.path.join(args.output_dir, "generalize_config.json"),
        {
            "holdout_drug_count": holdout_label,
            "train_val_pool_size": len(train_val_pool_df),
            "test_size": len(test_df),
        },
    )

    examples = data_lib.dataframe_to_examples(generalize_df)
    drug_to_idx, disease_to_idx = data_lib.build_mappings(examples)
    drug_idx_to_id = build_index_to_id(drug_to_idx, pad_token="<PAD>")
    disease_idx_to_id = build_index_to_id(disease_to_idx)

    if args.kg_embeddings:
        if node_ids is None or node_vectors is None:
            node_ids, node_vectors = kg_lib.load_precomputed_embeddings(
                args.kg_embeddings, args.kg_embedding_ids
            )
    else:
        node_ids, node_vectors = kg_lib.load_or_build_kg_embeddings(
            kg_path=args.kg,
            cache_path=args.kg_cache_path,
            embedding_dim=config["embedding_dim"],
            walk_length=config["kg_walk_length"],
            num_walks=config["kg_num_walks"],
            p=config["kg_p"],
            q=config["kg_q"],
            context_window=config["kg_context_window"],
            min_count=config["kg_min_count"],
            workers=config["kg_workers"],
            seed=config["seed"],
            src_col=args.edge_src_col,
            dst_col=args.edge_dst_col,
            edges=edges,
            backend=args.kg_backend,
        )

    embedding_dim = node_vectors.shape[1]
    if embedding_dim != config["embedding_dim"]:
        config["embedding_dim"] = embedding_dim

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    rng = np.random.RandomState(config["seed"])

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

    np.save(os.path.join(args.output_dir, "drug_embeddings.npy"), drug_embeddings)
    np.save(os.path.join(args.output_dir, "disease_embeddings.npy"), disease_embeddings)
    utils.save_json(
        os.path.join(args.output_dir, "drug_vocab.json"),
        {"ids": drug_idx_to_id},
    )
    utils.save_json(
        os.path.join(args.output_dir, "disease_vocab.json"),
        {"ids": disease_idx_to_id},
    )
    utils.save_json(os.path.join(args.output_dir, "config.json"), config)

    def encode_df(df: data_lib.pd.DataFrame) -> Tuple[list, list, list]:
        ex = data_lib.dataframe_to_examples(df)
        return data_lib.encode_examples(ex, drug_to_idx, disease_to_idx)

    train_seqs, train_diseases, train_labels = encode_df(train_df)
    val_seqs, val_diseases, val_labels = encode_df(val_df)
    test_seqs, test_diseases, test_labels = encode_df(test_df)

    train_dataset = data_lib.PolypharmacyDataset(
        train_seqs, train_diseases, train_labels
    )
    val_dataset = data_lib.PolypharmacyDataset(val_seqs, val_diseases, val_labels)
    test_dataset = data_lib.PolypharmacyDataset(test_seqs, test_diseases, test_labels)
    collate = lambda batch: data_lib.collate_batch(batch, pad_idx=0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_lib.PolypharmacyLSTMClassifier(
        drug_embeddings=torch.tensor(drug_embeddings),
        disease_embeddings=torch.tensor(disease_embeddings),
        lstm_hidden_dim=config["lstm_hidden_dim"],
        mlp_hidden_dim=config["mlp_hidden_dim"],
        mlp_layers=config["mlp_layers"],
        dropout=config["dropout"],
        freeze_kg=config["freeze_kg"],
        disease_token_position=config.get("disease_token_position"),
        concat_disease_after_lstm=config.get("concat_disease_after_lstm", True),
        pad_idx=0,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = float("-inf")
    best_path = os.path.join(args.output_dir, "best_model.pt")

    print(
        "Split sizes (train/val/test): "
        f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}"
    )

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for drug_seq, lengths, disease_idx, batch_labels in train_loader:
            drug_seq = drug_seq.to(device)
            lengths = lengths.to(device)
            disease_idx = disease_idx.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(drug_seq, lengths, disease_idx)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate_model(model, val_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))
        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.4f} | "
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
                    "drug_vocab_size": drug_embeddings.shape[0],
                    "disease_vocab_size": disease_embeddings.shape[0],
                    "embedding_dim": embedding_dim,
                    "lstm_hidden_dim": config["lstm_hidden_dim"],
                    "mlp_hidden_dim": config["mlp_hidden_dim"],
                    "mlp_layers": config["mlp_layers"],
                    "dropout": config["dropout"],
                    "freeze_kg": config["freeze_kg"],
                    "disease_token_position": config.get("disease_token_position"),
                    "concat_disease_after_lstm": config.get(
                        "concat_disease_after_lstm", True
                    ),
                    "pad_idx": 0,
                },
                best_path,
            )

    print(f"Best model saved to {best_path}")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        val_metrics = evaluate_model(model, val_loader, device)
    holdout_metrics = evaluate_model(model, test_loader, device)
    print(
        f"Test ({holdout_label} drugs) | auc={holdout_metrics['roc_auc']:.4f} | "
        f"acc={holdout_metrics['accuracy']:.4f} | sens={holdout_metrics['sensitivity']:.4f} | "
        f"spec={holdout_metrics['specificity']:.4f} | f1={holdout_metrics['f1']:.4f} | "
        f"confusion={holdout_metrics['confusion']}"
    )
    utils.save_json(
        os.path.join(args.output_dir, "generalize_metrics.json"),
        {
            "holdout_drug_count": holdout_label,
            "val_metrics": val_metrics,
            "test_metrics": holdout_metrics,
        },
    )


if __name__ == "__main__":
    main()
