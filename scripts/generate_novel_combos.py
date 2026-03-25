import argparse
import itertools
import math
import os
import time
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from polypharmacy import data as data_lib
from polypharmacy import model as model_lib
from polypharmacy import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and rank novel drug repurposing candidates using saved model artifacts."
    )
    parser.add_argument("--model-output-dir", required=True)
    parser.add_argument("--target-disease", required=True)
    parser.add_argument("--min-combo-size", type=int, default=2)
    parser.add_argument("--max-combo-size", type=int, default=2)
    parser.add_argument("--candidate-drugs-file", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-prob", type=float, default=0.9)
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument(
        "--novelty-source",
        choices=["filtered", "deduped"],
        default="deduped",
        help="Dataset source used to build known combos for novelty filtering.",
    )
    parser.add_argument("--output-dir", default="artifacts_repurpose")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.min_combo_size < 1:
        raise ValueError("--min-combo-size must be >= 1")
    if args.max_combo_size < args.min_combo_size:
        raise ValueError("--max-combo-size must be >= --min-combo-size")
    if args.max_candidates is not None and args.max_candidates < 1:
        raise ValueError("--max-candidates must be >= 1 when provided")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.min_prob < 0.0 or args.min_prob > 1.0:
        raise ValueError("--min-prob must be in [0, 1]")
    if args.top_percent <= 0.0 or args.top_percent > 100.0:
        raise ValueError("--top-percent must be in (0, 100]")
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")


def _canonical_combo(drug_ids: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(str(drug_id) for drug_id in drug_ids))


def _load_vocab_maps(model_output_dir: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    drug_vocab = utils.load_json(os.path.join(model_output_dir, "drug_vocab.json"))["ids"]
    disease_vocab = utils.load_json(os.path.join(model_output_dir, "disease_vocab.json"))["ids"]

    drug_to_idx = {
        str(drug_id): idx
        for idx, drug_id in enumerate(drug_vocab)
        if idx != 0 and isinstance(drug_id, str) and drug_id
    }
    disease_to_idx = {
        str(disease_id): idx
        for idx, disease_id in enumerate(disease_vocab)
        if isinstance(disease_id, str) and disease_id
    }
    return drug_to_idx, disease_to_idx


def _validate_target_disease(target_disease: str, disease_to_idx: Dict[str, int]) -> int:
    if not str(target_disease).startswith("MONDO:"):
        raise ValueError("--target-disease must be a MONDO ID (e.g., MONDO:0005148).")
    if target_disease not in disease_to_idx:
        raise ValueError(
            f"Target disease {target_disease} not found in disease_vocab.json."
        )
    return int(disease_to_idx[target_disease])


def _load_candidate_drugs(args: argparse.Namespace, drug_to_idx: Dict[str, int]) -> List[str]:
    if args.candidate_drugs_file is None:
        return sorted(drug_id for drug_id in drug_to_idx if drug_id.startswith("CHEBI:"))

    candidate_drugs: List[str] = []
    with open(args.candidate_drugs_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            drug_id = line.split("#", 1)[0].strip()
            if not drug_id:
                continue
            if not drug_id.startswith("CHEBI:"):
                continue
            if drug_id not in drug_to_idx:
                continue
            candidate_drugs.append(drug_id)
    return sorted(set(candidate_drugs))


def _load_known_combos_for_target_disease(
    model_output_dir: str, novelty_source: str, target_disease: str
) -> set:
    if novelty_source == "filtered":
        path = os.path.join(model_output_dir, "filtered_dataset_run.csv")
    else:
        path = os.path.join(model_output_dir, "deduped_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Novelty source file not found: {path}")

    df = pd.read_csv(path)
    if "drug_set" not in df.columns:
        if {"primary_drug_id_norm", "secondary_drug_id_norm"}.issubset(df.columns):
            primary = df["primary_drug_id_norm"].apply(data_lib.parse_list_column)
            secondary = df["secondary_drug_id_norm"].apply(data_lib.parse_list_column)
            df["drug_set"] = (primary + secondary).apply(data_lib.normalize_id_list)
        else:
            raise ValueError("Expected drug_set or primary/secondary drug columns in novelty source.")

    df["drug_set"] = df["drug_set"].apply(data_lib.parse_list_column)
    df["condition_id_norm"] = df["condition_id_norm"].astype(str)
    target_df = df[df["condition_id_norm"] == target_disease]
    return set(target_df["drug_set"].apply(_canonical_combo).tolist())


def _generate_candidate_combos(
    candidate_drugs: Sequence[str], min_combo_size: int, max_combo_size: int
) -> List[Tuple[str, ...]]:
    combos: List[Tuple[str, ...]] = []
    ordered_drugs = sorted(set(str(drug_id) for drug_id in candidate_drugs))
    for combo_size in range(min_combo_size, max_combo_size + 1):
        combos.extend(itertools.combinations(ordered_drugs, combo_size))
    return combos


def _select_max_candidates(
    combos: Sequence[Tuple[str, ...]], max_candidates: int, seed: int
) -> List[Tuple[str, ...]]:
    if len(combos) <= max_candidates:
        return list(combos)
    rng = np.random.RandomState(seed)
    picked = rng.choice(len(combos), size=max_candidates, replace=False)
    return [combos[idx] for idx in sorted(picked.tolist())]


def _load_model(model_output_dir: str) -> torch.nn.Module:
    checkpoint_path = os.path.join(model_output_dir, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    drug_embeddings = np.load(os.path.join(model_output_dir, "drug_embeddings.npy"))
    disease_embeddings = np.load(os.path.join(model_output_dir, "disease_embeddings.npy"))
    model = model_lib.PolypharmacyLSTMClassifier(
        drug_embeddings=torch.tensor(drug_embeddings),
        disease_embeddings=torch.tensor(disease_embeddings),
        lstm_hidden_dim=checkpoint["lstm_hidden_dim"],
        mlp_hidden_dim=checkpoint["mlp_hidden_dim"],
        mlp_layers=checkpoint.get("mlp_layers", 2),
        dropout=checkpoint["dropout"],
        freeze_kg=checkpoint["freeze_kg"],
        disease_token_position=checkpoint.get("disease_token_position"),
        concat_disease_after_lstm=checkpoint.get("concat_disease_after_lstm", True),
        pad_idx=checkpoint.get("pad_idx", 0),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _score_candidates(
    model: torch.nn.Module,
    candidate_combos: Sequence[Tuple[str, ...]],
    target_disease_idx: int,
    drug_to_idx: Dict[str, int],
    batch_size: int,
) -> np.ndarray:
    drug_sequences = [[drug_to_idx[drug_id] for drug_id in combo] for combo in candidate_combos]
    disease_indices = [target_disease_idx] * len(candidate_combos)
    dummy_labels = [0] * len(candidate_combos)
    dataset = data_lib.PolypharmacyDataset(drug_sequences, disease_indices, dummy_labels)
    collate = lambda batch: data_lib.collate_batch(batch, pad_idx=0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for drug_seq, lengths, disease_idx, _ in loader:
            logits = model(
                drug_seq.to(device),
                lengths.to(device),
                disease_idx.to(device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    if not all_probs:
        return np.array([], dtype=np.float32)
    return np.concatenate(all_probs)


def _rank_and_filter(
    candidate_combos: Sequence[Tuple[str, ...]],
    probs: np.ndarray,
    target_disease: str,
    model_output_dir: str,
    min_prob: float,
    top_percent: float,
    top_n: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if len(candidate_combos) != len(probs):
        raise ValueError("candidate_combos and probs length mismatch")

    df = pd.DataFrame(
        {
            "target_disease_id": [target_disease] * len(candidate_combos),
            "combo_tuple": list(candidate_combos),
            "combo_size": [len(combo) for combo in candidate_combos],
            "p_indication": probs.astype(float),
        }
    )
    df["uncertainty_margin"] = (df["p_indication"] - 0.5).abs()
    df["drug_set"] = df["combo_tuple"].apply(lambda combo: str(list(combo)))
    df["is_novel_per_disease"] = True
    df["model_output_dir"] = model_output_dir

    counts = {
        "num_scored": int(len(df)),
        "num_after_prob_filter": 0,
        "num_after_percent_filter": 0,
        "num_exported": 0,
    }

    df = df[df["p_indication"] >= min_prob].copy()
    counts["num_after_prob_filter"] = int(len(df))
    if len(df) == 0:
        return df, counts

    df = df.sort_values(
        by=["p_indication", "uncertainty_margin", "drug_set"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    keep_top_percent = max(1, int(math.ceil(len(df) * (top_percent / 100.0))))
    df = df.head(keep_top_percent).copy()
    counts["num_after_percent_filter"] = int(len(df))

    df = df.head(top_n).copy()
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    counts["num_exported"] = int(len(df))
    return df, counts


def main() -> None:
    start = time.time()
    args = parse_args()
    _validate_args(args)
    utils.set_seeds(args.seed)

    run_name = args.run_name or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    utils.ensure_dir(run_dir)

    drug_to_idx, disease_to_idx = _load_vocab_maps(args.model_output_dir)
    target_disease_idx = _validate_target_disease(args.target_disease, disease_to_idx)

    candidate_drugs = _load_candidate_drugs(args, drug_to_idx)
    if not candidate_drugs:
        raise ValueError("No candidate drugs available after filtering to known CHEBI IDs.")

    known_combos = _load_known_combos_for_target_disease(
        args.model_output_dir, args.novelty_source, args.target_disease
    )
    all_combos = _generate_candidate_combos(
        candidate_drugs, args.min_combo_size, args.max_combo_size
    )
    novel_combos = [combo for combo in all_combos if combo not in known_combos]
    removed_known = len(all_combos) - len(novel_combos)

    if args.max_candidates is not None:
        novel_combos = _select_max_candidates(novel_combos, args.max_candidates, args.seed)

    model = _load_model(args.model_output_dir)
    probs = _score_candidates(
        model=model,
        candidate_combos=novel_combos,
        target_disease_idx=target_disease_idx,
        drug_to_idx=drug_to_idx,
        batch_size=args.batch_size,
    )
    ranked_df, counts = _rank_and_filter(
        candidate_combos=novel_combos,
        probs=probs,
        target_disease=args.target_disease,
        model_output_dir=args.model_output_dir,
        min_prob=args.min_prob,
        top_percent=args.top_percent,
        top_n=args.top_n,
    )

    csv_path = os.path.join(run_dir, "ranked_candidates.csv")
    summary_path = os.path.join(run_dir, "summary.json")
    ranked_df.to_csv(csv_path, index=False)
    summary = {
        "target_disease_id": args.target_disease,
        "candidate_drug_count": int(len(candidate_drugs)),
        "min_combo_size": int(args.min_combo_size),
        "max_combo_size": int(args.max_combo_size),
        "num_generated_before_novelty_filter": int(len(all_combos)),
        "num_removed_known_for_disease": int(removed_known),
        "num_scored": counts["num_scored"],
        "num_after_prob_filter": counts["num_after_prob_filter"],
        "num_after_percent_filter": counts["num_after_percent_filter"],
        "num_exported": counts["num_exported"],
        "thresholds": {
            "min_prob": float(args.min_prob),
            "top_percent": float(args.top_percent),
            "top_n": int(args.top_n),
        },
        "artifact_paths": {
            "model_output_dir": args.model_output_dir,
            "checkpoint": os.path.join(args.model_output_dir, "best_model.pt"),
            "drug_vocab": os.path.join(args.model_output_dir, "drug_vocab.json"),
            "disease_vocab": os.path.join(args.model_output_dir, "disease_vocab.json"),
            "drug_embeddings": os.path.join(args.model_output_dir, "drug_embeddings.npy"),
            "disease_embeddings": os.path.join(args.model_output_dir, "disease_embeddings.npy"),
        },
        "seed": int(args.seed),
        "run_name": run_name,
        "output_csv": csv_path,
        "timing_seconds": float(time.time() - start),
    }
    utils.save_json(summary_path, summary)

    print(f"Wrote ranked candidates: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    print(
        "Counts | "
        f"generated={summary['num_generated_before_novelty_filter']} "
        f"novel_scored={summary['num_scored']} "
        f"exported={summary['num_exported']}"
    )


if __name__ == "__main__":
    main()
