from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from polypharmacy_kedro import data as data_lib
from polypharmacy_kedro import kg as kg_lib
from polypharmacy_kedro import model as model_lib
from polypharmacy_kedro import utils


def seed_everything(params: Dict[str, object]) -> None:
    utils.set_seeds(int(params["seed"]))


def build_deduped_dataset(
    indications_df: pd.DataFrame,
    contraindications_df: pd.DataFrame,
    params: Dict[str, object],
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    single_ind_path = params.get("single_therapy_indications_path")
    single_contra_path = params.get("single_therapy_contraindications_path")

    single_ind_df = pd.read_csv(single_ind_path) if single_ind_path else None
    single_contra_df = pd.read_csv(single_contra_path) if single_contra_path else None

    deduped_df, conflict_count = data_lib.load_deduped_dataframe_from_frames(
        indications_df=indications_df,
        contraindications_df=contraindications_df,
        single_therapy_indications_df=single_ind_df,
        single_therapy_contraindications_df=single_contra_df,
    )
    class_counts = deduped_df["label"].value_counts().to_dict()
    return deduped_df, {"conflict_count": int(conflict_count)}, class_counts


def required_nodes_from_deduped(deduped_df: pd.DataFrame) -> List[str]:
    required_drugs = set(
        itertools.chain.from_iterable(deduped_df["drug_set"])  # type: ignore[arg-type]
    )
    required_diseases = set(deduped_df["condition_id_norm"])
    return sorted(required_drugs.union(required_diseases))


def normalize_kg_edges(
    kg_edges_df: pd.DataFrame, params: Dict[str, object]
) -> pd.DataFrame:
    return kg_lib.normalize_edges_df(
        kg_edges_df,
        src_col=params.get("edge_src_col"),
        dst_col=params.get("edge_dst_col"),
    )


def build_kg_embeddings(
    edges_df: pd.DataFrame,
    required_nodes: Sequence[str],
    params: Dict[str, object],
) -> Tuple[List[str], List[str], np.ndarray]:
    embeddings_path = params.get("kg_embeddings_path")
    embedding_ids_path = params.get("kg_embedding_ids_path")

    if embeddings_path:
        node_ids, node_vectors = kg_lib.load_precomputed_embeddings(
            str(embeddings_path),
            str(embedding_ids_path) if embedding_ids_path else None,
        )
        kg_nodes = sorted(set(node_ids))
        print(
            "KG coverage filtering: using precomputed embedding node IDs "
            f"(nodes={len(kg_nodes)})"
        )
        return kg_nodes, node_ids, node_vectors

    edges = edges_df.copy()
    initial_edge_count = len(edges)
    expanded_nodes = set(required_nodes)
    hop_logs: List[Tuple[int, int, int]] = []
    truncated = False
    hop_expansion = int(params.get("kg_hop_expansion", 0))
    if hop_expansion > 0:
        expanded_nodes, hop_logs, truncated = kg_lib.expand_node_set(
            edges,
            expanded_nodes,
            hops=hop_expansion,
            max_nodes=params.get("kg_expansion_max_nodes"),
        )
        print(
            "KG hop expansion: "
            f"k={hop_expansion}, nodes: required={len(required_nodes)} -> expanded={len(expanded_nodes)}"
        )
        if params.get("kg_expansion_verbose"):
            for hop, new_nodes, cum_nodes in hop_logs:
                print(f"hop {hop}: +{new_nodes} nodes (cum {cum_nodes})")
            if truncated:
                print(
                    "KG hop expansion stopped early because node count exceeded "
                    f"{params.get('kg_expansion_max_nodes')}"
                )

    edges = kg_lib.prune_edges_to_nodes(edges, expanded_nodes)
    if len(edges) < initial_edge_count:
        print(
            f"KG node filtering: reduced edges from {initial_edge_count} to {len(edges)} "
            f"to cover {len(required_nodes)} dataset nodes"
        )

    max_edges = params.get("max_edges")
    if max_edges is not None and len(edges) > int(max_edges):
        edges = edges.sample(n=int(max_edges), random_state=int(params["seed"]))
        edges = edges.reset_index(drop=True)
        print(f"KG edge sampling: using {len(edges)} edges")

    kg_nodes = kg_lib.extract_kg_nodes(edges)

    kg_cache_path = str(params.get("kg_cache_path", "data/02_intermediate/kg_embeddings.npz"))
    utils.ensure_dir(str(Path(kg_cache_path).parent))
    node_ids, node_vectors = kg_lib.load_or_build_kg_embeddings(
        kg_path=str(params.get("kg_path", "data/01_raw/kg_edges.parquet")),
        cache_path=kg_cache_path,
        embedding_dim=int(params["embedding_dim"]),
        walk_length=int(params["kg_walk_length"]),
        num_walks=int(params["kg_num_walks"]),
        p=float(params["kg_p"]),
        q=float(params["kg_q"]),
        context_window=int(params["kg_context_window"]),
        min_count=int(params["kg_min_count"]),
        workers=int(params["kg_workers"]),
        seed=int(params["seed"]),
        src_col=None,
        dst_col=None,
        edges=edges,
        backend=str(params.get("kg_backend", "auto")),
    )

    return sorted(set(kg_nodes)), node_ids, node_vectors


def filter_by_kg_coverage(
    deduped_df: pd.DataFrame, kg_nodes: Iterable[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    return data_lib.filter_by_kg_coverage(deduped_df, kg_nodes)


def sample_run_df(run_df: pd.DataFrame, params: Dict[str, object]) -> pd.DataFrame:
    max_examples = params.get("max_examples")
    if max_examples is not None and len(run_df) > int(max_examples):
        run_df = run_df.sample(n=int(max_examples), random_state=int(params["seed"]))
        run_df = run_df.reset_index(drop=True)
        print(f"Example sampling: using {len(run_df)} examples")

    if len(run_df) == 0:
        raise ValueError(
            "No examples remain after KG coverage filtering. "
            "Increase --max-edges or use the full KG to improve coverage."
        )
    return run_df


def encode_dataset(
    run_df: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[str, int], List[List[int]], List[int], List[int]]:
    if len(run_df) > 0 and isinstance(run_df.iloc[0]["drug_set"], str):
        run_df = run_df.copy()
        run_df["drug_set"] = run_df["drug_set"].apply(data_lib.parse_list_column)
    examples = data_lib.dataframe_to_examples(run_df)
    drug_to_idx, disease_to_idx = data_lib.build_mappings(examples)
    drug_seqs, disease_idxs, labels = data_lib.encode_examples(
        examples, drug_to_idx, disease_to_idx
    )
    return drug_to_idx, disease_to_idx, drug_seqs, disease_idxs, labels


def make_splits(
    labels: Sequence[int], params: Dict[str, object]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx, test_idx = data_lib.deterministic_split(
        num_examples=len(labels),
        seed=int(params["seed"]),
        train_frac=float(params["train_frac"]),
        val_frac=float(params["val_frac"]),
        test_frac=float(params["test_frac"]),
    )
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Not enough examples after filtering to populate train/val/test splits; "
            "increase max_edges/max_examples or rerun without aggressive sampling."
        )
    print(
        "Dataset split sizes (train/val/test): "
        f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


def _build_index_to_id(mapping: Dict[str, int], pad_token: Optional[str] = None) -> List[str]:
    size = len(mapping) + (1 if pad_token is not None else 0)
    idx_to_id = ["" for _ in range(size)]
    if pad_token is not None:
        idx_to_id[0] = pad_token
    for entity_id, idx in mapping.items():
        idx_to_id[idx] = entity_id
    return idx_to_id


def build_entity_embeddings(
    node_ids: Sequence[str],
    node_vectors: np.ndarray,
    drug_to_idx: Dict[str, int],
    disease_to_idx: Dict[str, int],
    params: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]], Dict[str, List[str]], int]:
    embedding_dim = int(node_vectors.shape[1])
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    rng = np.random.RandomState(int(params["seed"]))

    drug_idx_to_id = _build_index_to_id(drug_to_idx, pad_token="<PAD>")
    disease_idx_to_id = _build_index_to_id(disease_to_idx)

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

    return (
        drug_embeddings,
        disease_embeddings,
        {"ids": drug_idx_to_id},
        {"ids": disease_idx_to_id},
        embedding_dim,
    )


def export_config(params: Dict[str, object], embedding_dim: int) -> Dict[str, object]:
    resolved = dict(params)
    resolved["embedding_dim"] = embedding_dim
    return resolved


def _evaluate_model(
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


def train_model(
    drug_seqs: Sequence[Sequence[int]],
    disease_idxs: Sequence[int],
    labels: Sequence[int],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    drug_embeddings: np.ndarray,
    disease_embeddings: np.ndarray,
    embedding_dim: int,
    params: Dict[str, object],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    dataset = data_lib.PolypharmacyDataset(drug_seqs, disease_idxs, labels)
    collate = lambda batch: data_lib.collate_batch(batch, pad_idx=0)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(params["batch_size"]),
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(params["batch_size"]),
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_lib.PolypharmacyLSTMClassifier(
        drug_embeddings=torch.tensor(drug_embeddings),
        disease_embeddings=torch.tensor(disease_embeddings),
        lstm_hidden_dim=int(params["lstm_hidden_dim"]),
        mlp_hidden_dim=int(params["mlp_hidden_dim"]),
        mlp_layers=int(params["mlp_layers"]),
        dropout=float(params["dropout"]),
        freeze_kg=bool(params["freeze_kg"]),
        disease_token_position=params.get("disease_token_position"),
        concat_disease_after_lstm=bool(params.get("concat_disease_after_lstm", True)),
        pad_idx=0,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = float("-inf")
    best_checkpoint: Dict[str, object] = {}
    best_metrics: Dict[str, object] = {}

    for epoch in range(1, int(params["epochs"]) + 1):
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

        val_metrics = _evaluate_model(model, val_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))
        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_sens={val_metrics['sensitivity']:.4f} | val_spec={val_metrics['specificity']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        if not np.isnan(val_metrics["roc_auc"]) and val_metrics["roc_auc"] > best_auc:
            best_auc = float(val_metrics["roc_auc"])
            best_metrics = {"epoch": epoch, **val_metrics}
            best_checkpoint = {
                "model_state": model.state_dict(),
                "config": dict(params),
                "drug_vocab_size": drug_embeddings.shape[0],
                "disease_vocab_size": disease_embeddings.shape[0],
                "embedding_dim": int(embedding_dim),
                "lstm_hidden_dim": int(params["lstm_hidden_dim"]),
                "mlp_hidden_dim": int(params["mlp_hidden_dim"]),
                "mlp_layers": int(params["mlp_layers"]),
                "dropout": float(params["dropout"]),
                "freeze_kg": bool(params["freeze_kg"]),
                "disease_token_position": params.get("disease_token_position"),
                "concat_disease_after_lstm": bool(
                    params.get("concat_disease_after_lstm", True)
                ),
                "pad_idx": 0,
            }

    return best_checkpoint, best_metrics


def evaluate_model(
    checkpoint: Dict[str, object],
    drug_seqs: Sequence[Sequence[int]],
    disease_idxs: Sequence[int],
    labels: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    drug_embeddings: np.ndarray,
    disease_embeddings: np.ndarray,
    params: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    dataset = data_lib.PolypharmacyDataset(drug_seqs, disease_idxs, labels)
    collate = lambda batch: data_lib.collate_batch(batch, pad_idx=0)
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(params.get("eval_batch_size", params["batch_size"])),
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=int(params.get("eval_batch_size", params["batch_size"])),
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_lib.PolypharmacyLSTMClassifier(
        drug_embeddings=torch.tensor(drug_embeddings),
        disease_embeddings=torch.tensor(disease_embeddings),
        lstm_hidden_dim=int(checkpoint["lstm_hidden_dim"]),
        mlp_hidden_dim=int(checkpoint["mlp_hidden_dim"]),
        mlp_layers=int(checkpoint.get("mlp_layers", 2)),
        dropout=float(checkpoint["dropout"]),
        freeze_kg=bool(checkpoint["freeze_kg"]),
        disease_token_position=checkpoint.get("disease_token_position"),
        concat_disease_after_lstm=bool(checkpoint.get("concat_disease_after_lstm", True)),
        pad_idx=int(checkpoint.get("pad_idx", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    val_metrics = _evaluate_model(model, val_loader, device)
    test_metrics = _evaluate_model(model, test_loader, device)

    return {"validation": val_metrics, "test": test_metrics}
