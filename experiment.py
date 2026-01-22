import argparse
import itertools
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from polypharmacy import config as config_lib
from polypharmacy import data as data_lib
from polypharmacy import kg as kg_lib
from polypharmacy import model as model_lib
from polypharmacy import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RF vs LSTM comparison.")
    parser.add_argument("--config", default=None, help="Optional JSON config overrides.")
    parser.add_argument("--output-dir", default="artifacts/experiment")
    parser.add_argument("--kg", default="kg_edges.parquet")
    parser.add_argument("--kg-hop-expansion", type=int, default=0)
    parser.add_argument("--kg-expansion-max-nodes", type=int, default=None)
    parser.add_argument("--kg-expansion-verbose", action="store_true")
    parser.add_argument("--kg-workers", type=int, default=None)
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
        "--single-therapy-indications",
        default=None,
        help="Optional RENCI single-therapy indications CSV.",
    )
    parser.add_argument(
        "--single-therapy-contraindications",
        default=None,
        help="Optional RENCI single-therapy contraindications CSV.",
    )
    return parser.parse_args()


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


def evaluate_model_with_loader(
    model: model_lib.PolypharmacyLSTMClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for drug_seq, lengths, disease_idx, labels in loader:
            drug_seq = drug_seq.to(device)
            lengths = lengths.to(device)
            disease_idx = disease_idx.to(device)
            logits = model(drug_seq, lengths, disease_idx)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    if not all_probs:
        return {"roc_auc": float("nan"), "accuracy": float("nan")}
    return utils.compute_metrics(np.concatenate(all_labels), np.concatenate(all_probs))


def train_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    drug_embeddings: np.ndarray,
    disease_embeddings: np.ndarray,
    device: torch.device,
    output_dir: str,
) -> Tuple[model_lib.PolypharmacyLSTMClassifier, str]:
    model = model_lib.PolypharmacyLSTMClassifier(
        drug_embeddings=torch.tensor(drug_embeddings),
        disease_embeddings=torch.tensor(disease_embeddings),
        lstm_hidden_dim=config["lstm_hidden_dim"],
        mlp_hidden_dim=config["mlp_hidden_dim"],
        mlp_layers=config["mlp_layers"],
        dropout=config["dropout"],
        freeze_kg=config["freeze_kg"],
        pad_idx=0,
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
                    "drug_vocab_size": drug_embeddings.shape[0],
                    "disease_vocab_size": disease_embeddings.shape[0],
                    "embedding_dim": drug_embeddings.shape[1],
                    "lstm_hidden_dim": config["lstm_hidden_dim"],
                    "mlp_hidden_dim": config["mlp_hidden_dim"],
                    "mlp_layers": config["mlp_layers"],
                    "dropout": config["dropout"],
                    "freeze_kg": config["freeze_kg"],
                    "pad_idx": 0,
                },
                best_path,
            )
    return model, best_path


def main() -> None:
    args = parse_args()
    config = config_lib.load_config(args.config)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dir(args.output_dir)
    utils.save_json(os.path.join(args.output_dir, "config.json"), config)

    deduped_df, conflict_count = data_lib.load_deduped_dataframe(
        "indications_norm.csv",
        "contraindications_norm.csv",
        single_therapy_indications_path=args.single_therapy_indications,
        single_therapy_contraindications_path=args.single_therapy_contraindications,
    )
    print(f"Conflict resolution: {conflict_count}")

    edges = kg_lib.load_edges(args.kg, src_col=None, dst_col=None)
    initial_edges = len(edges)
    required_drugs = set(
        itertools.chain.from_iterable(deduped_df["drug_set"])  # type: ignore[arg-type]
    )
    required_diseases = set(deduped_df["condition_id_norm"])
    required_nodes = required_drugs.union(required_diseases)

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
    )

    kg_nodes = kg_lib.extract_kg_nodes(pruned_edges)
    filtered_df, dropped_df, _ = data_lib.filter_by_kg_coverage(
        deduped_df, kg_nodes
    )
    ensure_dir(args.output_dir)
    filtered_df.to_csv(os.path.join(args.output_dir, "filtered_dataset.csv"), index=False)
    dropped_df.to_csv(os.path.join(args.output_dir, "dropped_rows.csv"), index=False)

    split_data = prepare_test_train_split(filtered_df, args.test_frac, args.seed, args.output_dir)
    train_df = split_data["train"]
    test_df = split_data["test"]

    stratify_col = train_df["label"] if train_df["label"].nunique() > 1 else None
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=args.seed,
        stratify=stratify_col,
    )

    filtered_examples = data_lib.dataframe_to_examples(filtered_df)
    drug_to_idx, disease_to_idx = data_lib.build_mappings(filtered_examples)
    drug_idx_to_id = build_idx_to_id(drug_to_idx, pad_token="<PAD>")
    disease_idx_to_id = build_idx_to_id(disease_to_idx)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    embedding_dim = node_vectors.shape[1]
    rng = np.random.RandomState(args.seed)
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

    def encode_split(df: data_lib.pd.DataFrame) -> Tuple[List[List[int]], List[int], List[int]]:
        examples = data_lib.dataframe_to_examples(df)
        return data_lib.encode_examples(examples, drug_to_idx, disease_to_idx)

    train_seqs, train_diseases, train_labels = encode_split(train_df)
    val_seqs, val_diseases, val_labels = encode_split(val_df)
    test_seqs, test_diseases, test_labels = encode_split(test_df)

    all_seqs, all_diseases, all_labels = data_lib.encode_examples(
        filtered_examples, drug_to_idx, disease_to_idx
    )
    train_idx, val_idx, test_idx = data_lib.deterministic_split(
        num_examples=len(all_labels),
        seed=config["seed"],
        train_frac=config["train_frac"],
        val_frac=config["val_frac"],
        test_frac=config["test_frac"],
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "splits.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_examples=len(all_labels),
    )
    filtered_df.to_csv(os.path.join(args.output_dir, "filtered_dataset_run.csv"), index=False)

    if not train_seqs or not val_seqs or not test_seqs:
        raise ValueError("One of the splits became empty after encoding.")

    train_dataset = data_lib.PolypharmacyDataset(train_seqs, train_diseases, train_labels)
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
    rf_model = fit_random_forest(
        X_rf_train,
        y_rf_train,
        est=args.rf_estimators,
        max_depth=args.rf_max_depth,
        seed=args.seed,
    )
    X_rf_test, y_rf_test = build_rf_features(
        test_df, drug_to_idx, disease_to_idx, drug_embeddings, disease_embeddings
    )
    rf_probs = rf_model.predict_proba(X_rf_test)[:, 1]
    rf_metrics = utils.compute_metrics(y_rf_test, rf_probs)
    print(
        f"RF on held-out 2-drug test | auc={rf_metrics['roc_auc']:.4f} | "
        f"acc={rf_metrics['accuracy']:.4f} | sens={rf_metrics['sensitivity']:.4f} | "
        f"spec={rf_metrics['specificity']:.4f} | f1={rf_metrics['f1']:.4f} | "
        f"confusion={rf_metrics['confusion']}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model, lstm_best_path = train_lstm(
        train_loader, val_loader, config, drug_embeddings, disease_embeddings, device, args.output_dir
    )
    lstm_metrics = evaluate_model_with_loader(lstm_model, test_loader, device)
    print(
        f"LSTM on held-out 2-drug test | auc={lstm_metrics['roc_auc']:.4f} | "
        f"acc={lstm_metrics['accuracy']:.4f} | sens={lstm_metrics['sensitivity']:.4f} | "
        f"spec={lstm_metrics['specificity']:.4f} | f1={lstm_metrics['f1']:.4f} | "
        f"confusion={lstm_metrics['confusion']}"
    )
    print(f"Best LSTM checkpoint saved to {lstm_best_path}")


if __name__ == "__main__":
    main()
