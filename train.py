import argparse
import itertools
import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from polypharmacy import config as config_lib
from polypharmacy import data as data_lib
from polypharmacy import kg as kg_lib
from polypharmacy import model as model_lib
from polypharmacy import utils


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
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return utils.compute_metrics(labels, probs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train polypharmacy predictor.")
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
        "--kg-backend",
        choices=["auto", "pecanpy", "node2vec"],
        default="auto",
        help="Force KG embedding backend.",
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
        "--quick",
        action="store_true",
        help="Run a quick smoke test with reduced data/compute.",
    )
    return parser.parse_args()


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
    for key, value in override_map.items():
        if value is not None:
            config[key] = value
    print("Resolved train config:\n" + json.dumps(config, indent=2))

    utils.set_seeds(config["seed"])
    utils.ensure_dir(args.output_dir)

    deduped_df, conflict_count = data_lib.load_deduped_dataframe(
        args.indications,
        args.contraindications,
        single_therapy_indications_path=args.single_therapy_indications,
        single_therapy_contraindications_path=args.single_therapy_contraindications,
    )
    print(f"Conflict resolution: {conflict_count} conflicting keys set to label=0")
    single_sources = []
    if args.single_therapy_indications:
        single_sources.append("single-therapy indications")
    if args.single_therapy_contraindications:
        single_sources.append("single-therapy contraindications")
    if single_sources:
        print(f"Including additional single-therapy data: {', '.join(single_sources)}")
    deduped_path = os.path.join(args.output_dir, "deduped_dataset.csv")
    deduped_df.to_csv(deduped_path, index=False)
    deduped_counts = deduped_df["label"].value_counts().to_dict()
    print(f"Deduped class counts: {deduped_counts}")

    required_drugs = set(
        itertools.chain.from_iterable(deduped_df["drug_set"])  # type: ignore[arg-type]
    )
    required_diseases = set(deduped_df["condition_id_norm"])
    required_nodes = required_drugs.union(required_diseases)

    edges = kg_lib.load_edges(args.kg, src_col=args.edge_src_col, dst_col=args.edge_dst_col)
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
    filtered_counts = filtered_df["label"].value_counts().to_dict()
    print(f"Filtered class counts: {filtered_counts}")
    run_df = filtered_df
    if args.max_examples is not None and len(filtered_df) > args.max_examples:
        run_df = filtered_df.sample(n=args.max_examples, random_state=config["seed"]).reset_index(
            drop=True
        )
        print(f"Example sampling: using {len(run_df)} examples")

    run_path = os.path.join(args.output_dir, "filtered_dataset_run.csv")
    run_df.to_csv(run_path, index=False)

    if len(run_df) == 0:
        raise ValueError(
            "No examples remain after KG coverage filtering. "
            "Increase --max-edges or use the full KG to improve coverage."
        )

    examples = data_lib.dataframe_to_examples(run_df)
    drug_to_idx, disease_to_idx = data_lib.build_mappings(examples)
    drug_seqs, disease_idxs, labels = data_lib.encode_examples(
        examples, drug_to_idx, disease_to_idx
    )

    splits_path = os.path.join(args.output_dir, "splits.npz")
    if os.path.exists(splits_path):
        split_data = np.load(splits_path)
        if split_data.get("num_examples", None) == len(labels):
            train_idx = split_data["train_idx"]
            val_idx = split_data["val_idx"]
            test_idx = split_data["test_idx"]
        else:
            train_idx, val_idx, test_idx = data_lib.deterministic_split(
                num_examples=len(labels),
                seed=config["seed"],
                train_frac=config["train_frac"],
                val_frac=config["val_frac"],
                test_frac=config["test_frac"],
            )
            np.savez_compressed(
                splits_path,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                num_examples=len(labels),
            )
    else:
        train_idx, val_idx, test_idx = data_lib.deterministic_split(
            num_examples=len(labels),
            seed=config["seed"],
            train_frac=config["train_frac"],
            val_frac=config["val_frac"],
            test_frac=config["test_frac"],
        )
        np.savez_compressed(
            splits_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            num_examples=len(labels),
        )

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Not enough examples after filtering to populate train/val/test splits; "
            "increase --max-edges/--max-examples or rerun without aggressive sampling."
        )

    print(
        f"Dataset split sizes (train/val/test): {len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
    )

    dataset = data_lib.PolypharmacyDataset(drug_seqs, disease_idxs, labels)
    collate = lambda batch: data_lib.collate_batch(batch, pad_idx=0)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )

    if args.kg_embeddings:
        node_ids, node_vectors = kg_lib.load_precomputed_embeddings(
            args.kg_embeddings, args.kg_embedding_ids
        )
    else:
        kg_cache_path = os.path.join(args.output_dir, "kg_embeddings.npz")
        node_ids, node_vectors = kg_lib.load_or_build_kg_embeddings(
            kg_path=args.kg,
            cache_path=kg_cache_path,
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

    drug_idx_to_id = build_index_to_id(drug_to_idx, pad_token="<PAD>")
    disease_idx_to_id = build_index_to_id(disease_to_idx)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = float("-inf")
    best_path = os.path.join(args.output_dir, "best_model.pt")

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
                    "pad_idx": 0,
                },
                best_path,
            )

    print(f"Best model saved to {best_path}")


if __name__ == "__main__":
    main()
