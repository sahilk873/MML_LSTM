#!/usr/bin/env python3
import argparse
import itertools
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from polypharmacy import data as data_lib
from polypharmacy import model as model_lib
from polypharmacy import utils


DEFAULT_IBD_SUBSTITUTION = {"MONDO:0011699": "MONDO:0005265"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank novel 2-drug CHEBI pairs across the disease panel using model vocab "
            "embeddings from a trained experiment artifact."
        )
    )
    parser.add_argument(
        "--model-type",
        choices=["rf", "pair_mlp"],
        default="pair_mlp",
        help="Pairwise model to use for scoring concatenated [drug1, drug2, disease] features.",
    )
    parser.add_argument("--model-output-dir", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--disease-reference-md", default="disease_codes_reference.md")
    parser.add_argument("--novelty-source", choices=["filtered", "deduped"], default="filtered")
    parser.add_argument("--candidate-drugs-file", default=None)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--output-dir", default="artifacts/vocab_repurpose")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


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


def _parse_mondo_codes_from_markdown(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    rows = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 4:
            continue
        disease_name = parts[1]
        match = re.search(r"(MONDO:\d+)", parts[2])
        if match:
            rows.append((disease_name, match.group(1)))
    if not rows:
        raise ValueError(f"No MONDO codes parsed from {path}")
    return rows


def _load_known_combos_for_target_disease(
    model_output_dir: str, novelty_source: str, target_disease: str
) -> Set[Tuple[str, ...]]:
    if novelty_source == "filtered":
        candidates = [
            os.path.join(model_output_dir, "filtered_dataset_run.csv"),
            os.path.join(model_output_dir, "filtered_dataset.csv"),
            os.path.join(model_output_dir, "deduped_dataset.csv"),
        ]
    else:
        candidates = [
            os.path.join(model_output_dir, "deduped_dataset.csv"),
            os.path.join(model_output_dir, "filtered_dataset_run.csv"),
            os.path.join(model_output_dir, "filtered_dataset.csv"),
        ]
    path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
    if path is None:
        raise FileNotFoundError(f"Novelty source file not found. Tried: {candidates}")

    df = pd.read_csv(path)
    if "drug_set" not in df.columns:
        if {"primary_drug_id_norm", "secondary_drug_id_norm"}.issubset(df.columns):
            primary = df["primary_drug_id_norm"].apply(data_lib.parse_list_column)
            secondary = df["secondary_drug_id_norm"].apply(data_lib.parse_list_column)
            df["drug_set"] = (primary + secondary).apply(data_lib.normalize_id_list)
        else:
            raise ValueError("Expected drug_set or primary/secondary drug columns in novelty source.")
    else:
        df["drug_set"] = df["drug_set"].apply(data_lib.parse_list_column)
    df["condition_id_norm"] = df["condition_id_norm"].astype(str)
    target_df = df[df["condition_id_norm"] == target_disease]
    return set(target_df["drug_set"].apply(_canonical_combo).tolist())


def _load_candidate_drugs(
    candidate_drugs_file: str | None, drug_to_idx: Dict[str, int]
) -> List[str]:
    if candidate_drugs_file is None:
        return sorted(drug_id for drug_id in drug_to_idx if drug_id.startswith("CHEBI:"))
    selected = []
    with open(candidate_drugs_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            drug_id = line.split("#", 1)[0].strip()
            if drug_id in drug_to_idx:
                selected.append(drug_id)
    return sorted(set(selected))


def _build_score_batch(model_type: str, model_path: str, sample_feature_dim: int):
    if model_type == "rf":
        import pickle

        with open(model_path, "rb") as handle:
            model = pickle.load(handle)
        if hasattr(model, "n_features_in_") and int(model.n_features_in_) != sample_feature_dim:
            raise ValueError(
                f"RF feature mismatch: model={model.n_features_in_}, expected={sample_feature_dim}"
            )

        def score_batch(X: np.ndarray) -> np.ndarray:
            return model.predict_proba(X)[:, 1].astype(np.float32)

        return score_batch

    checkpoint = torch.load(model_path, map_location="cpu")
    if int(checkpoint["input_dim"]) != sample_feature_dim:
        raise ValueError(
            f"PairMLP feature mismatch: model={checkpoint['input_dim']}, expected={sample_feature_dim}"
        )
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
            logits = model(torch.tensor(X, dtype=torch.float32))
            return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    return score_batch


def _score_pairs_for_disease(
    disease_code_reference: str,
    disease_code_used: str,
    disease_name: str,
    selected_drug_ids: Sequence[str],
    drug_to_idx: Dict[str, int],
    disease_to_idx: Dict[str, int],
    drug_embeddings: np.ndarray,
    disease_embeddings: np.ndarray,
    known_combos: Set[Tuple[str, ...]],
    score_batch,
    batch_size: int,
    top_n: int,
    score_column: str,
) -> pd.DataFrame:
    all_pairs = list(itertools.combinations(selected_drug_ids, 2))
    novel_pairs = [pair for pair in all_pairs if pair not in known_combos]
    if not novel_pairs:
        return pd.DataFrame(
            columns=[
                "rank",
                "disease_name",
                "disease_code_reference",
                "disease_code_used",
                "drug_id_1",
                "drug_id_2",
                score_column,
            ]
        )

    disease_vec = disease_embeddings[disease_to_idx[disease_code_used]]
    probs = np.empty(len(novel_pairs), dtype=np.float32)
    for start in range(0, len(novel_pairs), batch_size):
        chunk = novel_pairs[start : start + batch_size]
        X = np.stack(
            [
                np.concatenate(
                    (
                        drug_embeddings[drug_to_idx[drug_a]],
                        drug_embeddings[drug_to_idx[drug_b]],
                        disease_vec,
                    )
                )
                for drug_a, drug_b in chunk
            ],
            axis=0,
        )
        probs[start : start + len(chunk)] = score_batch(X)

    ranked_idx = np.argsort(-probs)[:top_n]
    rows = []
    for rank, idx in enumerate(ranked_idx, start=1):
        drug_a, drug_b = novel_pairs[int(idx)]
        rows.append(
            {
                "rank": rank,
                "disease_name": disease_name,
                "disease_code_reference": disease_code_reference,
                "disease_code_used": disease_code_used,
                "drug_id_1": drug_a,
                "drug_id_2": drug_b,
                score_column: float(probs[int(idx)]),
            }
        )
    return pd.DataFrame(rows)


def _summarize_combined_rankings(combined: pd.DataFrame) -> Dict[str, object]:
    if combined.empty:
        return {
            "exported_rows": 0,
            "unique_pairs": 0,
            "pairs_recommended_for_multiple_diseases": 0,
            "reused_pair_rows": 0,
        }
    pair_keys = combined.apply(
        lambda row: _canonical_combo((str(row["drug_id_1"]), str(row["drug_id_2"]))),
        axis=1,
    )
    pair_counts = pair_keys.value_counts()
    reused = pair_counts[pair_counts > 1]
    return {
        "exported_rows": int(len(combined)),
        "unique_pairs": int(pair_counts.shape[0]),
        "pairs_recommended_for_multiple_diseases": int(reused.shape[0]),
        "reused_pair_rows": int(reused.sum()),
        "mean_diseases_per_pair": float(pair_counts.mean()),
        "max_disease_reuse_for_single_pair": int(pair_counts.max()),
    }


def main() -> None:
    start = time.time()
    args = parse_args()
    utils.set_seeds(args.seed)

    run_name = args.run_name or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    utils.ensure_dir(run_dir)

    drug_to_idx, disease_to_idx = _load_vocab_maps(args.model_output_dir)
    drug_embeddings = np.load(os.path.join(args.model_output_dir, "drug_embeddings.npy"))
    disease_embeddings = np.load(os.path.join(args.model_output_dir, "disease_embeddings.npy"))
    selected_drug_ids = _load_candidate_drugs(args.candidate_drugs_file, drug_to_idx)
    if not selected_drug_ids:
        raise ValueError("No candidate drugs available after filtering.")

    disease_rows = _parse_mondo_codes_from_markdown(args.disease_reference_md)
    sample_drug = selected_drug_ids[0]
    sample_disease = DEFAULT_IBD_SUBSTITUTION.get(disease_rows[0][1], disease_rows[0][1])
    if sample_disease not in disease_to_idx:
        raise ValueError(f"Sample disease {sample_disease} not present in disease vocab.")
    sample_feature_dim = (
        int(drug_embeddings[drug_to_idx[sample_drug]].shape[0]) * 2
        + int(disease_embeddings[disease_to_idx[sample_disease]].shape[0])
    )
    model_path = args.model_path or os.path.join(
        args.model_output_dir,
        "pair_mlp_best.pt" if args.model_type == "pair_mlp" else "rf_model.pkl",
    )
    score_batch = _build_score_batch(args.model_type, model_path, sample_feature_dim)
    score_column = "pair_mlp_p_indication" if args.model_type == "pair_mlp" else "rf_p_indication"

    all_ranked = []
    summary_rows = []
    for disease_name, disease_code_reference in disease_rows:
        disease_code_used = DEFAULT_IBD_SUBSTITUTION.get(disease_code_reference, disease_code_reference)
        if disease_code_used not in disease_to_idx:
            summary_rows.append(
                {
                    "disease_name": disease_name,
                    "disease_code_reference": disease_code_reference,
                    "disease_code_used": disease_code_used,
                    "status": "skipped_missing_disease_vocab",
                    "num_generated_all_pairs": 0,
                    "num_novel_pairs": 0,
                    "num_exported": 0,
                }
            )
            continue

        known_combos = _load_known_combos_for_target_disease(
            args.model_output_dir, args.novelty_source, disease_code_used
        )
        ranked_df = _score_pairs_for_disease(
            disease_code_reference=disease_code_reference,
            disease_code_used=disease_code_used,
            disease_name=disease_name,
            selected_drug_ids=selected_drug_ids,
            drug_to_idx=drug_to_idx,
            disease_to_idx=disease_to_idx,
            drug_embeddings=drug_embeddings,
            disease_embeddings=disease_embeddings,
            known_combos=known_combos,
            score_batch=score_batch,
            batch_size=args.batch_size,
            top_n=args.top_n,
            score_column=score_column,
        )
        out_name = f"top{args.top_n}_{disease_code_used.replace(':', '_')}.csv"
        ranked_df.to_csv(os.path.join(run_dir, out_name), index=False)
        all_ranked.append(ranked_df)
        total_all = len(selected_drug_ids) * (len(selected_drug_ids) - 1) // 2
        summary_rows.append(
            {
                "disease_name": disease_name,
                "disease_code_reference": disease_code_reference,
                "disease_code_used": disease_code_used,
                "status": "ok",
                "num_generated_all_pairs": int(total_all),
                "num_novel_pairs": int(max(0, total_all - len(known_combos))),
                "num_exported": int(len(ranked_df)),
            }
        )

    combined = (
        pd.concat(all_ranked, ignore_index=True)
        if all_ranked
        else pd.DataFrame(
            columns=[
                "rank",
                "disease_name",
                "disease_code_reference",
                "disease_code_used",
                "drug_id_1",
                "drug_id_2",
                score_column,
            ]
        )
    )
    combined.to_csv(os.path.join(run_dir, f"top{args.top_n}_all_diseases.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(run_dir, "disease_run_summary.csv"), index=False)

    summary_payload = {
        "run_name": run_name,
        "model_type": args.model_type,
        "model_output_dir": args.model_output_dir,
        "model_path": model_path,
        "disease_reference_md": args.disease_reference_md,
        "novelty_source": args.novelty_source,
        "candidate_drug_count": int(len(selected_drug_ids)),
        "top_n": int(args.top_n),
        "score_column": score_column,
        "uniqueness_summary": _summarize_combined_rankings(combined),
        "timing_seconds": float(time.time() - start),
    }
    utils.save_json(os.path.join(run_dir, "summary.json"), summary_payload)
    print(f"Wrote run directory: {run_dir}")
    print(f"Wrote combined rankings: {os.path.join(run_dir, f'top{args.top_n}_all_diseases.csv')}")


if __name__ == "__main__":
    main()
