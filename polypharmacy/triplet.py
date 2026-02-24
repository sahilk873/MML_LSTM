import json
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset


REQUIRED_COLUMNS = [
    "drug_id_norm",
    "target_id_norm",
    "disease_id_norm",
    "label",
    "drug_embedding",
    "target_embedding",
    "disease_embedding",
]

# Columns required when evaluating (embeddings are loaded from the checkpoint, not the parquet).
EVAL_REQUIRED_COLUMNS = [
    "drug_id_norm",
    "target_id_norm",
    "disease_id_norm",
    "label",
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def iter_parquet_row_groups(path: str, columns: Sequence[str]) -> Iterator[pd.DataFrame]:
    parquet_file = pq.ParquetFile(path)
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=list(columns))
        yield table.to_pandas()


def validate_parquet_schema(path: str) -> None:
    present = set(pq.read_schema(path).names)
    missing = [col for col in REQUIRED_COLUMNS if col not in present]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def validate_eval_parquet_schema(path: str) -> None:
    present = set(pq.read_schema(path).names)
    missing = [col for col in EVAL_REQUIRED_COLUMNS if col not in present]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def build_embedding_map(
    path: str, id_col: str, embedding_col: str
) -> Dict[str, np.ndarray]:
    id_to_embedding: Dict[str, np.ndarray] = {}
    for chunk in iter_parquet_row_groups(path, [id_col, embedding_col]):
        ids = chunk[id_col].astype(str).tolist()
        vectors = chunk[embedding_col].tolist()
        for entity_id, vector in zip(ids, vectors):
            if entity_id in id_to_embedding:
                continue
            if vector is None:
                continue
            arr = np.asarray(vector, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(
                    f"Expected 1D embedding for {entity_id} in column {embedding_col}"
                )
            id_to_embedding[entity_id] = arr
    if not id_to_embedding:
        raise ValueError(f"No embeddings found for {id_col}/{embedding_col} in {path}")
    return id_to_embedding


def build_vocab_and_embedding_table(
    id_to_embedding: Dict[str, np.ndarray], seed: int
) -> Tuple[List[str], np.ndarray]:
    ids = sorted(id_to_embedding)
    dim = int(next(iter(id_to_embedding.values())).shape[0])
    rng = np.random.RandomState(seed)
    unk = rng.normal(loc=0.0, scale=0.01, size=(dim,)).astype(np.float32)
    table = np.zeros((len(ids) + 1, dim), dtype=np.float32)
    table[0] = unk
    for idx, entity_id in enumerate(ids, start=1):
        vector = id_to_embedding[entity_id]
        if vector.shape[0] != dim:
            raise ValueError(
                f"Inconsistent embedding dim for {entity_id}: {vector.shape[0]} != {dim}"
            )
        table[idx] = vector
    return ["<UNK>"] + ids, table


def ids_to_index_map(vocab: Sequence[str]) -> Dict[str, int]:
    return {entity_id: idx for idx, entity_id in enumerate(vocab)}


@dataclass
class EncodedTriplets:
    drug_idx: np.ndarray
    target_idx: np.ndarray
    disease_idx: np.ndarray
    labels: np.ndarray
    drug_ids: np.ndarray
    target_ids: np.ndarray
    disease_ids: np.ndarray


def encode_triplets_from_parquet(
    path: str,
    drug_to_idx: Dict[str, int],
    target_to_idx: Dict[str, int],
    disease_to_idx: Dict[str, int],
) -> EncodedTriplets:
    drug_indices: List[np.ndarray] = []
    target_indices: List[np.ndarray] = []
    disease_indices: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    drug_ids_all: List[np.ndarray] = []
    target_ids_all: List[np.ndarray] = []
    disease_ids_all: List[np.ndarray] = []

    for chunk in iter_parquet_row_groups(
        path, ["drug_id_norm", "target_id_norm", "disease_id_norm", "label"]
    ):
        drug_ids = chunk["drug_id_norm"].astype(str).to_numpy()
        target_ids = chunk["target_id_norm"].astype(str).to_numpy()
        disease_ids = chunk["disease_id_norm"].astype(str).to_numpy()
        y = chunk["label"].to_numpy(dtype=np.int64)

        drug_idx = np.asarray([drug_to_idx.get(x, 0) for x in drug_ids], dtype=np.int64)
        target_idx = np.asarray([target_to_idx.get(x, 0) for x in target_ids], dtype=np.int64)
        disease_idx = np.asarray([disease_to_idx.get(x, 0) for x in disease_ids], dtype=np.int64)

        drug_indices.append(drug_idx)
        target_indices.append(target_idx)
        disease_indices.append(disease_idx)
        labels.append(y)
        drug_ids_all.append(drug_ids)
        target_ids_all.append(target_ids)
        disease_ids_all.append(disease_ids)

    return EncodedTriplets(
        drug_idx=np.concatenate(drug_indices, axis=0),
        target_idx=np.concatenate(target_indices, axis=0),
        disease_idx=np.concatenate(disease_indices, axis=0),
        labels=np.concatenate(labels, axis=0),
        drug_ids=np.concatenate(drug_ids_all, axis=0),
        target_ids=np.concatenate(target_ids_all, axis=0),
        disease_ids=np.concatenate(disease_ids_all, axis=0),
    )


class TripletDataset(Dataset):
    def __init__(
        self,
        drug_idx: np.ndarray,
        target_idx: np.ndarray,
        disease_idx: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.drug_idx = drug_idx
        self.target_idx = target_idx
        self.disease_idx = disease_idx
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.drug_idx[idx], dtype=torch.long),
            torch.tensor(self.target_idx[idx], dtype=torch.long),
            torch.tensor(self.disease_idx[idx], dtype=torch.long),
            torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        )


class TripletLSTMClassifier(torch.nn.Module):
    def __init__(
        self,
        drug_embedding_table: np.ndarray,
        target_embedding_table: np.ndarray,
        disease_embedding_table: np.ndarray,
        lstm_hidden_dim: int,
        mlp_hidden_dim: int,
        mlp_layers: int,
        dropout: float,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.drug_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(drug_embedding_table), freeze=freeze_embeddings
        )
        self.target_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(target_embedding_table), freeze=freeze_embeddings
        )
        self.disease_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(disease_embedding_table), freeze=freeze_embeddings
        )

        input_dim = int(drug_embedding_table.shape[1])
        if int(target_embedding_table.shape[1]) != input_dim:
            raise ValueError("Drug and target embedding dimensions must match for LSTM input.")
        disease_dim = int(disease_embedding_table.shape[1])

        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
        )
        layers: List[torch.nn.Module] = []
        mlp_in = lstm_hidden_dim + disease_dim
        for _ in range(max(1, mlp_layers)):
            layers.append(torch.nn.Linear(mlp_in, mlp_hidden_dim))
            layers.append(torch.nn.LayerNorm(mlp_hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            mlp_in = mlp_hidden_dim
        layers.append(torch.nn.Linear(mlp_in, 1))
        self.classifier = torch.nn.Sequential(*layers)

    def forward(
        self, drug_idx: torch.Tensor, target_idx: torch.Tensor, disease_idx: torch.Tensor
    ) -> torch.Tensor:
        drug_emb = self.drug_embedding(drug_idx)
        target_emb = self.target_embedding(target_idx)
        disease_emb = self.disease_embedding(disease_idx)

        sequence = torch.stack([drug_emb, target_emb], dim=1)
        _, (h_n, _) = self.lstm(sequence)
        triplet_repr = h_n[-1]
        combined = torch.cat([triplet_repr, disease_emb], dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits


def run_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for drug_idx, target_idx, disease_idx, labels in loader:
            drug_idx = drug_idx.to(device)
            target_idx = target_idx.to(device)
            disease_idx = disease_idx.to(device)
            logits = model(drug_idx, target_idx, disease_idx)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy().astype(np.int64))
    if not all_probs:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_classification_metrics(
    labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    metrics: Dict[str, float] = {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def aggregate_pair_predictions(
    prediction_df: pd.DataFrame,
    pair_cols: Sequence[str] = ("drug_id_norm", "disease_id_norm"),
) -> pd.DataFrame:
    grouped = (
        prediction_df.groupby(list(pair_cols), as_index=False)
        .agg(
            score=("score", "mean"),
            label=("label", "max"),
            n_triplets=("label", "size"),
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def compute_enrichment_factors(
    labels: np.ndarray, scores: np.ndarray, fracs: Iterable[float]
) -> Dict[str, Dict[str, float]]:
    labels = labels.astype(np.int64)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    total_n = int(labels_sorted.shape[0])
    positives_all = int(labels_sorted.sum())
    global_hit_rate = float(positives_all / total_n) if total_n > 0 else float("nan")
    output: Dict[str, Dict[str, float]] = {}

    for frac in fracs:
        key = f"ef_{int(round(frac * 100))}"
        if total_n == 0 or positives_all == 0:
            output[key] = {
                "fraction": float(frac),
                "k": 0,
                "positives_top": 0,
                "positives_all": positives_all,
                "hit_rate_top": float("nan"),
                "hit_rate_all": global_hit_rate,
                "ef": float("nan"),
            }
            continue
        k = max(1, int(np.floor(frac * total_n)))
        positives_top = int(labels_sorted[:k].sum())
        hit_rate_top = float(positives_top / k)
        ef = float(hit_rate_top / global_hit_rate)
        output[key] = {
            "fraction": float(frac),
            "k": int(k),
            "positives_top": positives_top,
            "positives_all": positives_all,
            "hit_rate_top": hit_rate_top,
            "hit_rate_all": global_hit_rate,
            "ef": ef,
        }
    return output
