#!/usr/bin/env python3
import argparse
import ast
import itertools
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from polypharmacy import data as data_lib
from polypharmacy import utils


DEFAULT_IBD_SUBSTITUTION = {"MONDO:0011699": "MONDO:0005265"}
PREFIX_PRIORITY = {
    "CHEBI": 0,
    "UNII": 1,
    "DRUGBANK": 2,
    "PUBCHEM.COMPOUND": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map MeDIC drugs via curie+alternate_ids to model vocab and rank all "
            "2-drug pairs for the disease set using saved RF model."
        )
    )
    parser.add_argument(
        "--model-output-dir",
        default="artifacts/exp_refined_mixed_twosides_topological512",
    )
    parser.add_argument("--rf-model-path", default=None)
    parser.add_argument(
        "--precomputed-node-ids",
        default="artifacts/precomputed_embeddings/topological/node_ids.npy",
    )
    parser.add_argument(
        "--precomputed-embeddings",
        default="artifacts/precomputed_embeddings/topological/embeddings.npy",
    )
    parser.add_argument("--medic-drug-list", default="MeDIC Drug List.csv")
    parser.add_argument(
        "--alias-index",
        default="artifacts/precomputed_embeddings/topological/equivalent_id_to_node_id.parquet",
    )
    parser.add_argument("--disease-reference-md", default="disease_codes_reference.md")
    parser.add_argument("--novelty-source", choices=["filtered", "deduped"], default="deduped")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=200000)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="artifacts/rf_repurpose_top50")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _canonical_combo(drug_ids: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(str(drug_id) for drug_id in drug_ids))


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
        code_field = parts[2]
        match = re.search(r"(MONDO:\d+)", code_field)
        if not match:
            continue
        rows.append((disease_name, match.group(1)))
    if not rows:
        raise ValueError(f"No MONDO codes parsed from {path}")
    return rows


def _parse_optional_id_list(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none"}:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                values = parsed
            else:
                values = [parsed]
        except (ValueError, SyntaxError):
            values = [token.strip().strip("'") for token in text.strip("[]").split(",")]
    else:
        values = [value]
    result = []
    for item in values:
        token = str(item).strip()
        if not token or token.lower() in {"nan", "none"}:
            continue
        if token.startswith("Error") or token.startswith("['Error"):
            continue
        result.append(token)
    return result


def _id_priority(identifier: str) -> Tuple[int, str]:
    prefix = str(identifier).split(":", 1)[0]
    return PREFIX_PRIORITY.get(prefix, 99), str(identifier)


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
        raise FileNotFoundError(
            f"Novelty source file not found. Tried: {candidates}"
        )

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


def _map_medic_drugs(
    medic_csv: str,
    alias_index_parquet: str,
    allowed_node_ids: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    allowed_set = set(allowed_node_ids) if allowed_node_ids is not None else None
    alias_df = pd.read_parquet(alias_index_parquet, columns=["alias_id", "node_id"])
    alias_to_nodes: Dict[str, Set[str]] = {}
    for row in alias_df.itertuples(index=False):
        alias = str(row.alias_id).strip()
        node = str(row.node_id).strip()
        if allowed_set is not None and node not in allowed_set:
            continue
        alias_to_nodes.setdefault(alias, set()).add(node)

    medic = pd.read_csv(medic_csv)
    required_cols = {"curie", "alternate_ids", "drug_name"}
    missing = required_cols - set(medic.columns)
    if missing:
        raise ValueError(f"Missing columns in {medic_csv}: {sorted(missing)}")

    matched_rows: List[Dict[str, object]] = []
    unmatched_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(medic.itertuples(index=False)):
        curie = str(row.curie).strip()
        alternates = _parse_optional_id_list(row.alternate_ids)
        candidate_aliases = []
        if curie:
            candidate_aliases.append(curie)
        candidate_aliases.extend([alt for alt in alternates if alt != curie])

        mapped_nodes: Set[str] = set()
        matched_aliases: List[str] = []
        for alias in candidate_aliases:
            nodes = alias_to_nodes.get(alias)
            if not nodes:
                continue
            mapped_nodes.update(nodes)
            matched_aliases.append(alias)

        if not mapped_nodes:
            unmatched_rows.append(
                {
                    "medic_row_index": idx,
                    "drug_name": str(row.drug_name),
                    "curie": curie,
                    "alternate_ids": str(row.alternate_ids),
                }
            )
            continue

        best_node = sorted(mapped_nodes, key=_id_priority)[0]
        matched_rows.append(
            {
                "medic_row_index": idx,
                "drug_name": str(row.drug_name),
                "curie": curie,
                "alternate_ids": str(row.alternate_ids),
                "selected_node_id": best_node,
                "selected_prefix": best_node.split(":", 1)[0],
                "matched_aliases": str(sorted(set(matched_aliases))),
                "num_candidate_node_ids": len(mapped_nodes),
            }
        )

    matched_df = pd.DataFrame(matched_rows).drop_duplicates(subset=["selected_node_id"])
    unmatched_df = pd.DataFrame(unmatched_rows)
    return matched_df, unmatched_df


def _build_index_for_required_ids(
    node_ids: np.ndarray, required_ids: Set[str]
) -> Dict[str, int]:
    required = set(str(value) for value in required_ids)
    out: Dict[str, int] = {}
    if not required:
        return out
    for idx, node in enumerate(node_ids):
        node_id = str(node)
        if node_id in required:
            out[node_id] = int(idx)
            if len(out) == len(required):
                break
    return out


def _build_drug_name_map(matched_df: pd.DataFrame) -> Dict[str, str]:
    required_cols = {"selected_node_id", "drug_name"}
    missing = required_cols - set(matched_df.columns)
    if missing:
        raise ValueError(f"Cannot build drug name map, missing columns: {sorted(missing)}")
    working = matched_df[["selected_node_id", "drug_name", "medic_row_index"]].copy()
    working["selected_node_id"] = working["selected_node_id"].astype(str)
    working["drug_name"] = working["drug_name"].astype(str).str.strip()
    working = working[working["drug_name"] != ""]
    working = working.sort_values(["selected_node_id", "medic_row_index"])
    working = working.drop_duplicates(subset=["selected_node_id"], keep="first")
    return working.set_index("selected_node_id")["drug_name"].to_dict()


def _attach_drug_names(df: pd.DataFrame, drug_name_map: Dict[str, str]) -> pd.DataFrame:
    if df.empty:
        for column in ("drug_name_1", "drug_name_2"):
            if column not in df.columns:
                df[column] = pd.Series(dtype="object")
        return df
    out = df.copy()
    for column in ("drug_name_1", "drug_name_2"):
        if column in out.columns:
            out = out.drop(columns=[column])
    out.insert(out.columns.get_loc("drug_id_1") + 1, "drug_name_1", out["drug_id_1"].map(drug_name_map))
    out.insert(out.columns.get_loc("drug_id_2") + 1, "drug_name_2", out["drug_id_2"].map(drug_name_map))
    return out


def _score_pairs_for_disease(
    disease_code: str,
    disease_name: str,
    disease_used: str,
    selected_drug_ids: Sequence[str],
    known_combos: Set[Tuple[str, ...]],
    node_to_idx: Dict[str, int],
    node_embeddings: np.ndarray,
    rf_model: object,
    top_n: int,
    batch_size: int,
) -> pd.DataFrame:
    all_combos = list(itertools.combinations(sorted(set(selected_drug_ids)), 2))
    novel_combos = [combo for combo in all_combos if _canonical_combo(combo) not in known_combos]
    if not novel_combos:
        return pd.DataFrame(
            columns=[
                "rank",
                "disease_name",
                "disease_code_reference",
                "disease_code_used",
                "drug_id_1",
                "drug_id_2",
                "rf_p_indication",
            ]
        )

    disease_idx = node_to_idx[disease_used]
    disease_vec = node_embeddings[disease_idx]
    probs = np.empty(len(novel_combos), dtype=np.float32)

    for start in range(0, len(novel_combos), batch_size):
        end = min(len(novel_combos), start + batch_size)
        feats = []
        for d1, d2 in novel_combos[start:end]:
            emb = np.concatenate(
                (
                    node_embeddings[node_to_idx[d1]],
                    node_embeddings[node_to_idx[d2]],
                    disease_vec,
                )
            )
            feats.append(emb)
        X = np.stack(feats, axis=0)
        probs[start:end] = rf_model.predict_proba(X)[:, 1].astype(np.float32)

    ranked_idx = np.argsort(-probs)[:top_n]
    rows = []
    for rank, idx in enumerate(ranked_idx, start=1):
        d1, d2 = novel_combos[int(idx)]
        rows.append(
            {
                "rank": rank,
                "disease_name": disease_name,
                "disease_code_reference": disease_code,
                "disease_code_used": disease_used,
                "drug_id_1": d1,
                "drug_id_2": d2,
                "rf_p_indication": float(probs[int(idx)]),
            }
        )
    return pd.DataFrame(rows)


def _run_single_disease(
    disease_name: str,
    disease_code: str,
    model_output_dir: str,
    novelty_source: str,
    selected_drug_ids: Sequence[str],
    node_to_idx: Dict[str, int],
    node_embeddings: np.ndarray,
    rf_model: object,
    top_n: int,
    batch_size: int,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, str]]:
    disease_used = DEFAULT_IBD_SUBSTITUTION.get(disease_code, disease_code)
    substitution_applied: Dict[str, str] = {}
    if disease_used != disease_code:
        substitution_applied[disease_code] = disease_used

    if disease_used not in node_to_idx:
        return (
            pd.DataFrame(
                columns=[
                    "rank",
                    "disease_name",
                    "disease_code_reference",
                    "disease_code_used",
                    "drug_id_1",
                    "drug_id_2",
                    "rf_p_indication",
                ]
            ),
            {
                "disease_name": disease_name,
                "disease_code_reference": disease_code,
                "disease_code_used": disease_used,
                "status": "skipped_missing_disease_vocab",
                "num_generated_all_pairs": 0,
                "num_novel_pairs": 0,
                "num_exported": 0,
            },
            substitution_applied,
        )

    known_combos = _load_known_combos_for_target_disease(
        model_output_dir, novelty_source, disease_used
    )
    total_all = len(selected_drug_ids) * (len(selected_drug_ids) - 1) // 2
    novel_estimate = total_all - len(known_combos)
    ranked_df = _score_pairs_for_disease(
        disease_code=disease_code,
        disease_name=disease_name,
        disease_used=disease_used,
        selected_drug_ids=selected_drug_ids,
        known_combos=known_combos,
        node_to_idx=node_to_idx,
        node_embeddings=node_embeddings,
        rf_model=rf_model,
        top_n=top_n,
        batch_size=batch_size,
    )
    summary_row = {
        "disease_name": disease_name,
        "disease_code_reference": disease_code,
        "disease_code_used": disease_used,
        "status": "ok",
        "num_generated_all_pairs": int(total_all),
        "num_novel_pairs": int(max(0, novel_estimate)),
        "num_exported": int(len(ranked_df)),
    }
    return ranked_df, summary_row, substitution_applied


def main() -> None:
    start = time.time()
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    utils.set_seeds(args.seed)

    run_name = args.run_name or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    utils.ensure_dir(run_dir)

    rf_model_path = args.rf_model_path or os.path.join(args.model_output_dir, "rf_model.pkl")
    with open(rf_model_path, "rb") as handle:
        rf_model = pickle.load(handle)

    node_ids = np.load(args.precomputed_node_ids, allow_pickle=True)
    node_embeddings = np.load(args.precomputed_embeddings, mmap_mode="r")
    if len(node_ids) != int(node_embeddings.shape[0]):
        raise ValueError(
            "Precomputed node_ids and embeddings length mismatch: "
            f"{len(node_ids)} vs {node_embeddings.shape[0]}"
        )

    disease_rows = _parse_mondo_codes_from_markdown(args.disease_reference_md)
    disease_used_codes = {
        DEFAULT_IBD_SUBSTITUTION.get(disease_code, disease_code)
        for _, disease_code in disease_rows
    }

    matched_df, unmatched_df = _map_medic_drugs(
        medic_csv=args.medic_drug_list,
        alias_index_parquet=args.alias_index,
        allowed_node_ids=None,
    )
    selected_drug_ids_raw = matched_df["selected_node_id"].astype(str).tolist()
    required_ids = set(selected_drug_ids_raw).union(disease_used_codes)
    node_to_idx = _build_index_for_required_ids(node_ids=node_ids, required_ids=required_ids)

    matched_df["in_precomputed_embeddings"] = matched_df["selected_node_id"].astype(str).isin(
        node_to_idx
    )
    selected_drug_ids = (
        matched_df[matched_df["in_precomputed_embeddings"]]["selected_node_id"].astype(str).tolist()
    )
    drug_name_map = _build_drug_name_map(matched_df)
    matched_df.to_csv(os.path.join(run_dir, "medic_mapping_matched.csv"), index=False)
    unmatched_df.to_csv(os.path.join(run_dir, "medic_mapping_unmatched.csv"), index=False)

    # Validate RF input dimensionality against concatenated embeddings.
    if not selected_drug_ids:
        raise ValueError("No MeDIC drugs mapped to precomputed embeddings.")
    sample_drug = selected_drug_ids[0]
    sample_disease = next(iter(disease_used_codes))
    if sample_disease not in node_to_idx:
        raise ValueError(
            f"Sample disease {sample_disease} not found in precomputed embeddings; "
            "cannot validate RF feature dimensions."
        )
    sample_feature_dim = (
        int(node_embeddings[node_to_idx[sample_drug]].shape[0]) * 2
        + int(node_embeddings[node_to_idx[sample_disease]].shape[0])
    )
    if hasattr(rf_model, "n_features_in_") and int(rf_model.n_features_in_) != sample_feature_dim:
        raise ValueError(
            "RF model feature dimension does not match precomputed embeddings: "
            f"model={rf_model.n_features_in_}, expected={sample_feature_dim}"
        )
    # Avoid oversubscribing CPU cores when parallelizing across diseases.
    if hasattr(rf_model, "n_jobs"):
        rf_model.n_jobs = 1 if args.max_workers > 1 else rf_model.n_jobs

    substitution_applied: Dict[str, str] = {}
    all_ranked = []
    summary_rows = []
    worker_count = min(args.max_workers, max(1, len(disease_rows)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(
            executor.map(
                lambda pair: _run_single_disease(
                    disease_name=pair[0],
                    disease_code=pair[1],
                    model_output_dir=args.model_output_dir,
                    novelty_source=args.novelty_source,
                    selected_drug_ids=selected_drug_ids,
                    node_to_idx=node_to_idx,
                    node_embeddings=node_embeddings,
                    rf_model=rf_model,
                    top_n=args.top_n,
                    batch_size=args.batch_size,
                ),
                disease_rows,
            )
        )

    for ranked_df, summary_row, substitution_map in results:
        substitution_applied.update(substitution_map)
        disease_used = str(summary_row["disease_code_used"])
        out_name = f"top50_{disease_used.replace(':', '_')}.csv"
        ranked_with_names = _attach_drug_names(ranked_df, drug_name_map)
        ranked_with_names.to_csv(os.path.join(run_dir, out_name), index=False)
        all_ranked.append(ranked_df)
        summary_rows.append(summary_row)

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
                "rf_p_indication",
            ]
        )
    )
    combined = _attach_drug_names(combined, drug_name_map)
    combined.to_csv(os.path.join(run_dir, "top50_all_diseases.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(run_dir, "disease_run_summary.csv"), index=False)

    summary_payload = {
        "run_name": run_name,
        "model_output_dir": args.model_output_dir,
        "rf_model_path": rf_model_path,
        "alias_index": args.alias_index,
        "precomputed_node_ids": args.precomputed_node_ids,
        "precomputed_embeddings": args.precomputed_embeddings,
        "medic_drug_list": args.medic_drug_list,
        "disease_reference_md": args.disease_reference_md,
        "novelty_source": args.novelty_source,
        "top_n": args.top_n,
        "max_workers": args.max_workers,
        "mapped_medic_drugs": int(len(matched_df)),
        "mapped_medic_drugs_in_precomputed": int(len(selected_drug_ids)),
        "unmatched_medic_drugs": int(len(unmatched_df)),
        "ibd_substitutions": substitution_applied,
        "timing_seconds": float(time.time() - start),
    }
    utils.save_json(os.path.join(run_dir, "summary.json"), summary_payload)

    print(f"Wrote run directory: {run_dir}")
    print(
        "Mapping coverage | "
        f"matched={len(matched_df)} unmatched={len(unmatched_df)} "
        f"total={len(matched_df) + len(unmatched_df)}"
    )
    print(f"Wrote combined rankings: {os.path.join(run_dir, 'top50_all_diseases.csv')}")


if __name__ == "__main__":
    main()
