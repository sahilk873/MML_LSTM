import ast
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LIST_COLUMNS = ("primary_drug_id_norm", "secondary_drug_id_norm")
SINGLE_THERAPY_DRUG_COLUMN = "final normalized drug id"
SINGLE_THERAPY_DISEASE_COLUMN = "final normalized disease id"
TWOSIDES_DRUG_1_COLUMN = "drug_1_rxnorn_id_norm"
TWOSIDES_DRUG_2_COLUMN = "drug_2_rxnorm_id_norm"
TWOSIDES_DISEASE_COLUMN = "condition_meddra_id_norm"


def _parse_list_column(value: object) -> List[str]:
    """Parse list-valued columns stored as stringified Python lists."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none"}:
            return []
        # CSV typically stores list-like strings (e.g., "['CHEBI:6413']"), but some
        # deduplicated exports contain bare IDs (e.g., "CHEBI:6413"). Accept both.
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, str):
                return [parsed]
        except (SyntaxError, ValueError):
            return [text]
    raise ValueError(f"Unable to parse list column value: {value!r}")


def parse_list_column(value: object) -> List[str]:
    """Public wrapper for list parsing to keep reuse consistent."""
    return normalize_id_list(_parse_list_column(value))


def normalize_id_list(values: List[object]) -> List[str]:
    """Flatten one level and coerce identifiers to strings."""
    flattened: List[str] = []

    def _valid_id(text: str) -> bool:
        return bool(text.strip()) and not text.strip().startswith("Error")

    for item in values:
        if item is None or (isinstance(item, float) and np.isnan(item)):
            continue
        if isinstance(item, list):
            for nested in item:
                if nested is None or (isinstance(nested, float) and np.isnan(nested)):
                    continue
                token = str(nested)
                if _valid_id(token):
                    flattened.append(token)
        else:
            token = str(item)
            if _valid_id(token):
                flattened.append(token)
    return flattened


def _is_invalid_identifier(value: object) -> bool:
    text = str(value).strip()
    if not text:
        return True
    lower = text.lower()
    if lower in {"nan", "none"}:
        return True
    if text.startswith("Error") or lower.startswith("error"):
        return True
    if text.startswith("['Error") or text.startswith('["Error'):
        return True
    return False


def _canonical_key(drug_set: Sequence[str], condition_id: str) -> Tuple[Tuple[str, ...], str]:
    return tuple(sorted(str(drug_id) for drug_id in drug_set)), str(condition_id)


@dataclass
class LabeledExample:
    drug_ids: List[str]
    disease_id: str
    label: int


def _load_csv_df(path: str, label: int, source_name: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in LIST_COLUMNS:
        df[column] = df[column].apply(parse_list_column)
    df["condition_id_norm"] = df["condition_id_norm"].astype(str)
    df["drug_set"] = (df["primary_drug_id_norm"] + df["secondary_drug_id_norm"]).apply(
        lambda ids: sorted(normalize_id_list(ids))
    )
    df = df[df["drug_set"].map(len) > 0]
    df = df[~df["condition_id_norm"].apply(_is_invalid_identifier)]
    df = df[["drug_set", "condition_id_norm"]].copy()
    # Label is inferred from the file source (indications=1, contraindications=0).
    df["label"] = label
    if source_name:
        df["source_name"] = source_name
    return df


def _parse_single_therapy_drug_ids(value: object) -> List[str]:
    """Split the single-therapy drug column into a normalized ID list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    tokens = [token.strip() for token in text.split("|") if token.strip()]
    tokens = [token for token in tokens if token.lower() not in {"nan", "none"}]
    return normalize_id_list(tokens)


def _load_single_therapy_csv(
    path: str, label: int, source_name: Optional[str] = None
) -> pd.DataFrame:
    """Load the RENCI single-therapy CSVs and align them with the standard schema."""
    df = pd.read_csv(path, usecols=[SINGLE_THERAPY_DRUG_COLUMN, SINGLE_THERAPY_DISEASE_COLUMN])
    df = df.rename(
        columns={
            SINGLE_THERAPY_DRUG_COLUMN: "drug_id",
            SINGLE_THERAPY_DISEASE_COLUMN: "condition_id_norm",
        }
    )
    df["drug_set"] = df["drug_id"].apply(_parse_single_therapy_drug_ids)
    df["condition_id_norm"] = df["condition_id_norm"].astype(str).str.strip()
    df = df[df["drug_set"].map(len) > 0]
    invalid_condition = df["condition_id_norm"].apply(_is_invalid_identifier)
    df = df[~invalid_condition]
    df = df[["drug_set", "condition_id_norm"]].copy()
    df["drug_set"] = df["drug_set"].apply(lambda ids: sorted(ids))
    df["label"] = label
    if source_name:
        df["source_name"] = source_name
    return df


def _load_twosides_csv(path: str, source_name: str = "twosides") -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[TWOSIDES_DRUG_1_COLUMN, TWOSIDES_DRUG_2_COLUMN, TWOSIDES_DISEASE_COLUMN],
    )
    df[TWOSIDES_DRUG_1_COLUMN] = df[TWOSIDES_DRUG_1_COLUMN].astype(str).str.strip()
    df[TWOSIDES_DRUG_2_COLUMN] = df[TWOSIDES_DRUG_2_COLUMN].astype(str).str.strip()
    df[TWOSIDES_DISEASE_COLUMN] = df[TWOSIDES_DISEASE_COLUMN].astype(str).str.strip()

    rows: List[Dict[str, object]] = []
    for row in df.itertuples(index=False):
        drug_1 = getattr(row, TWOSIDES_DRUG_1_COLUMN)
        drug_2 = getattr(row, TWOSIDES_DRUG_2_COLUMN)
        disease = getattr(row, TWOSIDES_DISEASE_COLUMN)
        if (
            _is_invalid_identifier(drug_1)
            or _is_invalid_identifier(drug_2)
            or _is_invalid_identifier(disease)
        ):
            continue
        drug_set = sorted(normalize_id_list([drug_1, drug_2]))
        if not drug_set:
            continue
        rows.append(
            {
                "drug_set": drug_set,
                "condition_id_norm": str(disease),
                "label": 0,
                "source_name": source_name,
            }
        )
    return pd.DataFrame(rows, columns=["drug_set", "condition_id_norm", "label", "source_name"])


def _build_randomized_disease_shuffle_negatives(
    sourced_negative_df: pd.DataFrame,
    positive_df: pd.DataFrame,
    known_keys: set,
    ratio: float,
    seed: int,
    source_name: str = "randomized_disease_shuffle",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    target = int(round(max(0.0, ratio) * len(sourced_negative_df)))
    stats: Dict[str, int] = {
        "target": target,
        "attempted": 0,
        "accepted": 0,
        "rejected_known_key": 0,
        "rejected_same_disease": 0,
    }
    if target <= 0 or sourced_negative_df.empty or positive_df.empty:
        return (
            pd.DataFrame(columns=["drug_set", "condition_id_norm", "label", "source_name"]),
            stats,
        )

    disease_pool = (
        positive_df["condition_id_norm"]
        .astype(str)
        .apply(str.strip)
        .loc[lambda s: ~s.apply(_is_invalid_identifier)]
        .unique()
    )
    if len(disease_pool) == 0:
        return (
            pd.DataFrame(columns=["drug_set", "condition_id_norm", "label", "source_name"]),
            stats,
        )

    rng = np.random.RandomState(seed)
    source_rows = sourced_negative_df.reset_index(drop=True)
    accepted: List[Dict[str, object]] = []
    max_attempts = max(100, target * 30)

    while len(accepted) < target and stats["attempted"] < max_attempts:
        row = source_rows.iloc[int(rng.randint(len(source_rows)))]
        random_disease = str(disease_pool[int(rng.randint(len(disease_pool)))])
        stats["attempted"] += 1
        if random_disease == str(row.condition_id_norm):
            stats["rejected_same_disease"] += 1
            continue
        key = _canonical_key(row.drug_set, random_disease)
        if key in known_keys:
            stats["rejected_known_key"] += 1
            continue
        known_keys.add(key)
        accepted.append(
            {
                "drug_set": list(row.drug_set),
                "condition_id_norm": random_disease,
                "label": 0,
                "source_name": source_name,
            }
        )

    stats["accepted"] = len(accepted)
    return (
        pd.DataFrame(accepted, columns=["drug_set", "condition_id_norm", "label", "source_name"]),
        stats,
    )


def load_deduped_dataframe(
    indications_path: str,
    contraindications_path: str,
    single_therapy_indications_path: Optional[str] = None,
    single_therapy_contraindications_path: Optional[str] = None,
    twosides_contraindications_path: Optional[str] = None,
    enable_mixed_negatives: bool = False,
    random_negative_ratio: float = 1.0,
    random_negative_strategy: str = "disease_shuffle",
    seed: int = 13,
    report_out: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Load CSVs, build sorted drug sets, and deduplicate with conflict resolution."""
    positive_frames = [
        _load_csv_df(indications_path, label=1, source_name="indications"),
    ]
    negative_frames = [
        _load_csv_df(contraindications_path, label=0, source_name="contraindications"),
    ]
    if single_therapy_indications_path:
        positive_frames.append(
            _load_single_therapy_csv(
                single_therapy_indications_path,
                label=1,
                source_name="single_therapy_indications",
            )
        )
    if single_therapy_contraindications_path:
        negative_frames.append(
            _load_single_therapy_csv(
                single_therapy_contraindications_path,
                label=0,
                source_name="single_therapy_contraindications",
            )
        )
    if enable_mixed_negatives and twosides_contraindications_path:
        negative_frames.append(
            _load_twosides_csv(twosides_contraindications_path, source_name="twosides")
        )

    positives_df = pd.concat(positive_frames, ignore_index=True)
    sourced_negatives_df = pd.concat(negative_frames, ignore_index=True)
    combined = pd.concat([positives_df, sourced_negatives_df], ignore_index=True)

    random_stats: Dict[str, int] = {
        "target": 0,
        "attempted": 0,
        "accepted": 0,
        "rejected_known_key": 0,
        "rejected_same_disease": 0,
    }
    if enable_mixed_negatives:
        if random_negative_strategy != "disease_shuffle":
            raise ValueError(
                f"Unsupported random_negative_strategy={random_negative_strategy!r}. "
                "Supported strategies: disease_shuffle"
            )
        known_keys = {
            _canonical_key(row.drug_set, row.condition_id_norm)
            for row in combined.itertuples(index=False)
        }
        randomized_df, random_stats = _build_randomized_disease_shuffle_negatives(
            sourced_negatives_df,
            positives_df,
            known_keys=known_keys,
            ratio=random_negative_ratio,
            seed=seed,
        )
        if not randomized_df.empty:
            combined = pd.concat([combined, randomized_df], ignore_index=True)

    combined["drug_set_key"] = combined["drug_set"].apply(tuple)
    conflict_flags = (
        combined.groupby(["drug_set_key", "condition_id_norm"])["label"]
        .nunique()
        .reset_index(name="label_count")
    )
    conflict_count = int((conflict_flags["label_count"] > 1).sum())

    deduped = (
        combined.groupby(["drug_set_key", "condition_id_norm"], as_index=False)
        .agg(
            label=("label", "min"),
            drug_set=("drug_set", "first"),
            source_name=(
                "source_name",
                lambda values: "|".join(sorted({str(value) for value in values})),
            ),
        )
        .drop(columns=["drug_set_key"])
    )
    if report_out is not None:
        report_out.clear()
        report_out.update(
            {
                "enable_mixed_negatives": bool(enable_mixed_negatives),
                "random_negative_ratio": float(random_negative_ratio),
                "random_negative_strategy": str(random_negative_strategy),
                "source_counts_before_dedup": {
                    str(key): int(value)
                    for key, value in combined["source_name"].value_counts().to_dict().items()
                },
                "label_counts_before_dedup": {
                    int(key): int(value)
                    for key, value in combined["label"].value_counts().to_dict().items()
                },
                "source_counts_after_dedup": {
                    str(key): int(value)
                    for key, value in deduped["source_name"]
                    .str.split("|", regex=False)
                    .explode()
                    .value_counts()
                    .to_dict()
                    .items()
                },
                "label_counts_after_dedup": {
                    int(key): int(value)
                    for key, value in deduped["label"].value_counts().to_dict().items()
                },
                "random_negative_generation": random_stats,
                "conflict_count": int(conflict_count),
            }
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
    single_therapy_indications_path: Optional[str] = None,
    single_therapy_contraindications_path: Optional[str] = None,
    twosides_contraindications_path: Optional[str] = None,
    enable_mixed_negatives: bool = False,
    random_negative_ratio: float = 1.0,
    random_negative_strategy: str = "disease_shuffle",
    seed: int = 13,
) -> List[LabeledExample]:
    """Load labeled examples with deduplication and conflict resolution applied."""
    deduped, _ = load_deduped_dataframe(
        indications_path,
        contraindications_path,
        single_therapy_indications_path,
        single_therapy_contraindications_path,
        twosides_contraindications_path,
        enable_mixed_negatives,
        random_negative_ratio,
        random_negative_strategy,
        seed,
    )
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
    if num_examples <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    rng = np.random.RandomState(seed)
    indices = np.arange(num_examples)
    rng.shuffle(indices)
    if num_examples < 3:
        train_end = 1
        val_end = min(num_examples, 2)
    else:
        train_end = max(1, int(num_examples * train_frac))
        val_end = min(num_examples, train_end + max(1, int(num_examples * val_frac)))
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
