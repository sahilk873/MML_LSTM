#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from polypharmacy import model as model_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score exact drug-drug-disease triples with saved RF, LSTM, and pair MLP models."
    )
    parser.add_argument("--triples-json", required=True)
    parser.add_argument("--model-output-dir", required=True)
    parser.add_argument(
        "--precomputed-node-ids",
        default="artifacts/precomputed_embeddings/topological/node_ids.npy",
    )
    parser.add_argument(
        "--precomputed-embeddings",
        default="artifacts/precomputed_embeddings/topological/embeddings.npy",
    )
    parser.add_argument(
        "--alias-index",
        default="artifacts/precomputed_embeddings/topological/equivalent_id_to_node_id.parquet",
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_triples(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("triples-json must contain a JSON list.")
    return payload


def build_alias_map(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    import pandas as pd

    df = pd.read_parquet(path, columns=["alias_id", "node_id"])
    alias_to_node: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        alias = str(row.alias_id).strip()
        node = str(row.node_id).strip()
        alias_to_node.setdefault(alias, node)
    return alias_to_node


def resolve_entity_id(entity_id: str, node_to_idx: Dict[str, int], alias_to_node: Dict[str, str]) -> str:
    if entity_id in node_to_idx:
        return entity_id
    return alias_to_node.get(entity_id, entity_id)


def build_pair_models(model_output_dir: str):
    with open(os.path.join(model_output_dir, "rf_model.pkl"), "rb") as handle:
        rf_model = pickle.load(handle)
    if hasattr(rf_model, "n_jobs"):
        rf_model.n_jobs = 1

    checkpoint = torch.load(
        os.path.join(model_output_dir, "pair_mlp_best.pt"), map_location="cpu"
    )
    pair_model = model_lib.PairEmbeddingMLPClassifier(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        init_sigma=None,
    )
    pair_model.load_state_dict(checkpoint["model_state"])
    pair_model.eval()
    return rf_model, pair_model


def build_lstm_model(model_output_dir: str):
    checkpoint = torch.load(os.path.join(model_output_dir, "best_model.pt"), map_location="cpu")
    if checkpoint.get("uses_direct_embeddings"):
        model = model_lib.PolypharmacyDirectEmbeddingLSTMClassifier(
            drug_embedding_dim=int(checkpoint["drug_embedding_dim"]),
            disease_embedding_dim=int(checkpoint["disease_embedding_dim"]),
            lstm_hidden_dim=int(checkpoint["lstm_hidden_dim"]),
            mlp_hidden_dim=int(checkpoint["mlp_hidden_dim"]),
            mlp_layers=int(checkpoint.get("mlp_layers", 2)),
            dropout=float(checkpoint["dropout"]),
            disease_token_position=checkpoint.get("disease_token_position"),
            concat_disease_after_lstm=bool(checkpoint.get("concat_disease_after_lstm", True)),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, None, None, True

    drug_embeddings = np.load(os.path.join(model_output_dir, "drug_embeddings.npy"))
    disease_embeddings = np.load(os.path.join(model_output_dir, "disease_embeddings.npy"))
    with open(os.path.join(model_output_dir, "drug_vocab.json"), "r", encoding="utf-8") as handle:
        drug_vocab = json.load(handle)["ids"]
    with open(
        os.path.join(model_output_dir, "disease_vocab.json"), "r", encoding="utf-8"
    ) as handle:
        disease_vocab = json.load(handle)["ids"]

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
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, {str(x): i for i, x in enumerate(drug_vocab)}, {
        str(x): i for i, x in enumerate(disease_vocab)
    }, False


def score_pair_models(
    drug_ids: List[str],
    disease_id: str,
    node_to_idx: Dict[str, int],
    node_embeddings: np.ndarray,
    rf_model,
    pair_model,
):
    missing = [entity_id for entity_id in [*drug_ids, disease_id] if entity_id not in node_to_idx]
    if missing:
        return {"rf": None, "mlp": None, "missing": missing}
    d1, d2 = sorted(drug_ids)
    features = np.concatenate(
        [
            np.asarray(node_embeddings[node_to_idx[d1]]),
            np.asarray(node_embeddings[node_to_idx[d2]]),
            np.asarray(node_embeddings[node_to_idx[disease_id]]),
        ],
        axis=0,
    ).astype(np.float32)[None, :]
    rf_score = float(rf_model.predict_proba(features)[:, 1][0])
    with torch.no_grad():
        mlp_score = float(torch.sigmoid(pair_model(torch.tensor(features))).cpu().numpy()[0])
    return {"rf": rf_score, "mlp": mlp_score, "missing": []}


def score_lstm(
    drug_ids: List[str],
    disease_id: str,
    model: torch.nn.Module,
    drug_to_idx: Dict[str, int] | None,
    disease_to_idx: Dict[str, int] | None,
    use_direct_embeddings: bool,
    node_to_idx: Dict[str, int],
    node_embeddings: np.ndarray,
):
    if use_direct_embeddings:
        missing = [entity_id for entity_id in [*drug_ids, disease_id] if entity_id not in node_to_idx]
        if missing:
            return {"lstm": None, "missing": missing}
        ordered_drugs = sorted(drug_ids)
        drug_tensor = torch.tensor(
            [[np.asarray(node_embeddings[node_to_idx[drug_id]], dtype=np.float32) for drug_id in ordered_drugs]],
            dtype=torch.float32,
        )
        length_tensor = torch.tensor([len(ordered_drugs)], dtype=torch.long)
        disease_tensor = torch.tensor(
            [np.asarray(node_embeddings[node_to_idx[disease_id]], dtype=np.float32)],
            dtype=torch.float32,
        )
        with torch.no_grad():
            score = float(torch.sigmoid(model(drug_tensor, length_tensor, disease_tensor)).cpu().numpy()[0])
        return {"lstm": score, "missing": []}

    missing = [drug_id for drug_id in drug_ids if drug_to_idx.get(drug_id, 0) == 0]
    if disease_id not in disease_to_idx:
        missing.append(disease_id)
    if missing:
        return {"lstm": None, "missing": missing}
    ordered_drugs = sorted(drug_ids)
    drug_tensor = torch.tensor([[drug_to_idx[drug_id] for drug_id in ordered_drugs]], dtype=torch.long)
    length_tensor = torch.tensor([len(ordered_drugs)], dtype=torch.long)
    disease_tensor = torch.tensor([disease_to_idx[disease_id]], dtype=torch.long)
    with torch.no_grad():
        score = float(torch.sigmoid(model(drug_tensor, length_tensor, disease_tensor)).cpu().numpy()[0])
    return {"lstm": score, "missing": []}


def main() -> None:
    args = parse_args()
    triples = load_triples(args.triples_json)

    node_ids = np.load(args.precomputed_node_ids, allow_pickle=True)
    node_embeddings = np.load(args.precomputed_embeddings, mmap_mode="r")
    node_to_idx = {str(node_id): idx for idx, node_id in enumerate(node_ids.tolist())}

    alias_to_node = build_alias_map(args.alias_index)
    rf_model, pair_model = build_pair_models(args.model_output_dir)
    lstm_model, drug_to_idx, disease_to_idx, use_direct_embeddings = build_lstm_model(
        args.model_output_dir
    )

    results = []
    for triple in triples:
        label = str(triple["label"])
        original_drug_ids = [str(drug_id) for drug_id in triple["drug_ids"]]
        original_disease_id = str(triple["disease_id"])
        drug_ids = [
            resolve_entity_id(drug_id, node_to_idx, alias_to_node) for drug_id in original_drug_ids
        ]
        disease_id = resolve_entity_id(original_disease_id, node_to_idx, alias_to_node)
        pair_scores = score_pair_models(
            drug_ids, disease_id, node_to_idx, node_embeddings, rf_model, pair_model
        )
        lstm_scores = score_lstm(
            drug_ids,
            disease_id,
            lstm_model,
            drug_to_idx,
            disease_to_idx,
            use_direct_embeddings,
            node_to_idx,
            node_embeddings,
        )
        results.append(
            {
                "label": label,
                "drug_ids": original_drug_ids,
                "resolved_drug_ids": drug_ids,
                "disease_id": original_disease_id,
                "resolved_disease_id": disease_id,
                "rf": pair_scores["rf"],
                "mlp": pair_scores["mlp"],
                "lstm": lstm_scores["lstm"],
                "pair_missing": pair_scores["missing"],
                "lstm_missing": lstm_scores["missing"],
            }
        )

    output = json.dumps(results, indent=2)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
