import ast
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LIST_COLUMNS = ("primary_drug_id_norm", "secondary_drug_id_norm")


def _parse_list_column(value: object) -> List[str]:
    """Parse list-valued columns stored as stringified Python lists."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # CSV stores list-like strings (e.g., "['CHEBI:6413']") that need literal_eval.
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Unable to parse list column value: {value!r}")


def parse_list_column(value: object) -> List[str]:
    """Public wrapper for list parsing to keep reuse consistent."""
    return _parse_list_column(value)


@dataclass
class LabeledExample:
    drug_ids: List[str]
    disease_id: str
    label: int


def _load_csv_df(path: str, label: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in LIST_COLUMNS:
        df[column] = df[column].apply(_parse_list_column)
    df["condition_id_norm"] = df["condition_id_norm"].astype(str)
    df["drug_set"] = (df["primary_drug_id_norm"] + df["secondary_drug_id_norm"]).apply(
        lambda ids: sorted(ids)
    )
    df = df[df["drug_set"].map(len) > 0]
    df = df[~df["condition_id_norm"].isin(["nan", "None"])]
    df = df[["drug_set", "condition_id_norm"]].copy()
    # Label is inferred from the file source (indications=1, contraindications=0).
    df["label"] = label
    return df


def load_deduped_dataframe(
    indications_path: str,
    contraindications_path: str,
) -> Tuple[pd.DataFrame, int]:
    """Load CSVs, build sorted drug sets, and deduplicate with conflict resolution."""
    indications = _load_csv_df(indications_path, label=1)
    contraindications = _load_csv_df(contraindications_path, label=0)
    combined = pd.concat([indications, contraindications], ignore_index=True)

    combined["drug_set_key"] = combined["drug_set"].apply(tuple)
    conflict_flags = (
        combined.groupby(["drug_set_key", "condition_id_norm"])["label"]
        .nunique()
        .reset_index(name="label_count")
    )
    conflict_count = int((conflict_flags["label_count"] > 1).sum())

    deduped = (
        combined.groupby(["drug_set_key", "condition_id_norm"], as_index=False)
        .agg(label=("label", "min"), drug_set=("drug_set", "first"))
        .drop(columns=["drug_set_key"])
    )
    return deduped, conflict_count


def dataframe_to_examples(df: pd.DataFrame) -> List[LabeledExample]:
    examples: List[LabeledExample] = []
    for row in df.itertuples(index=False):
        examples.append(
            LabeledExample(
                drug_ids=list(row.drug_set),
                disease_id=row.condition_id_norm,
                label=int(row.label),
            )
        )
    return examples


def load_examples(
    indications_path: str,
    contraindications_path: str,
) -> List[LabeledExample]:
    """Load labeled examples with deduplication and conflict resolution applied."""
    deduped, _ = load_deduped_dataframe(indications_path, contraindications_path)
    return dataframe_to_examples(deduped)


def filter_by_kg_coverage(
    df: pd.DataFrame, kg_nodes: Iterable[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    kg_node_set = set(kg_nodes)
    dropped_rows = []
    kept_rows = []
    missing_prefixes = Counter()

    for row in df.itertuples(index=False):
        missing_drugs = [drug for drug in row.drug_set if drug not in kg_node_set]
        missing_disease = (
            row.condition_id_norm if row.condition_id_norm not in kg_node_set else None
        )
        if missing_drugs or missing_disease:
            reason = []
            if missing_drugs:
                reason.append("missing_drug")
            if missing_disease:
                reason.append("missing_disease")
            for missing_id in missing_drugs:
                prefix = missing_id.split(":", 1)[0]
                missing_prefixes[prefix] += 1
            if missing_disease:
                prefix = missing_disease.split(":", 1)[0]
                missing_prefixes[prefix] += 1
            dropped_rows.append(
                {
                    "drug_set": row.drug_set,
                    "condition_id_norm": row.condition_id_norm,
                    "label": row.label,
                    "missing_drug_ids": missing_drugs,
                    "missing_disease_id": missing_disease,
                    "reason": "+".join(reason),
                }
            )
        else:
            kept_rows.append(
                {
                    "drug_set": row.drug_set,
                    "condition_id_norm": row.condition_id_norm,
                    "label": row.label,
                }
            )

    filtered_df = pd.DataFrame(kept_rows, columns=["drug_set", "condition_id_norm", "label"])
    dropped_df = pd.DataFrame(
        dropped_rows,
        columns=[
            "drug_set",
            "condition_id_norm",
            "label",
            "missing_drug_ids",
            "missing_disease_id",
            "reason",
        ],
    )
    stats = {
        "num_dropped": len(dropped_rows),
        "percent_dropped": float(len(dropped_rows) / max(len(df), 1)),
        "missing_prefixes": missing_prefixes.most_common(10),
    }
    return filtered_df, dropped_df, stats


def build_vocab(values: Iterable[str]) -> List[str]:
    """Create a deterministic vocabulary list."""
    return sorted(set(values))


def build_mappings(examples: Sequence[LabeledExample]) -> Tuple[Dict[str, int], Dict[str, int]]:
    drug_ids: List[str] = []
    disease_ids: List[str] = []
    for example in examples:
        drug_ids.extend(example.drug_ids)
        disease_ids.append(example.disease_id)

    drug_vocab = build_vocab(drug_ids)
    disease_vocab = build_vocab(disease_ids)

    drug_to_idx = {drug_id: idx + 1 for idx, drug_id in enumerate(drug_vocab)}
    disease_to_idx = {disease_id: idx for idx, disease_id in enumerate(disease_vocab)}
    return drug_to_idx, disease_to_idx


def encode_examples(
    examples: Sequence[LabeledExample],
    drug_to_idx: Dict[str, int],
    disease_to_idx: Dict[str, int],
) -> Tuple[List[List[int]], List[int], List[int]]:
    drug_sequences: List[List[int]] = []
    disease_indices: List[int] = []
    labels: List[int] = []
    for example in examples:
        drug_sequences.append([drug_to_idx[drug_id] for drug_id in example.drug_ids])
        disease_indices.append(disease_to_idx[example.disease_id])
        labels.append(example.label)
    return drug_sequences, disease_indices, labels


def deterministic_split(
    num_examples: int,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Fixed seed + deterministic permutation keeps splits stable across runs.
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")
    rng = np.random.RandomState(seed)
    indices = np.arange(num_examples)
    rng.shuffle(indices)
    train_end = int(num_examples * train_frac)
    val_end = train_end + int(num_examples * val_frac)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx


class PolypharmacyDataset(Dataset):
    def __init__(
        self,
        drug_sequences: Sequence[Sequence[int]],
        disease_indices: Sequence[int],
        labels: Sequence[int],
    ) -> None:
        self.drug_sequences = drug_sequences
        self.disease_indices = disease_indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drug_seq = torch.tensor(self.drug_sequences[idx], dtype=torch.long)
        disease_idx = torch.tensor(self.disease_indices[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return drug_seq, disease_idx, label


def collate_batch(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    drug_seqs, disease_idxs, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in drug_seqs], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(
        drug_seqs, batch_first=True, padding_value=pad_idx
    )
    disease_tensor = torch.stack(disease_idxs)
    label_tensor = torch.stack(labels)
    return padded, lengths, disease_tensor, label_tensor
