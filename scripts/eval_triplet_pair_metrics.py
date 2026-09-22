import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from polypharmacy import triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate triplet LSTM baseline at triplet and pair levels.")
    parser.add_argument("--test-parquet", default="test_kushal.parquet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts_triplet_lstm")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ef-fracs", default="0.05,0.10,0.20")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=None)
    return parser.parse_args()


def parse_ef_fracs(raw: str) -> List[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0 or value >= 1:
            raise ValueError("EF fractions must be between 0 and 1.")
        values.append(value)
    if not values:
        raise ValueError("At least one EF fraction must be provided.")
    return values


def main() -> None:
    args = parse_args()
    triplet.set_all_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    triplet.validate_eval_parquet_schema(args.test_parquet)
    ef_fracs = parse_ef_fracs(args.ef_fracs)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    drug_vocab = checkpoint["drug_vocab"]
    target_vocab = checkpoint["target_vocab"]
    disease_vocab = checkpoint["disease_vocab"]
    drug_to_idx = triplet.ids_to_index_map(drug_vocab)
    target_to_idx = triplet.ids_to_index_map(target_vocab)
    disease_to_idx = triplet.ids_to_index_map(disease_vocab)

    encoded = triplet.encode_triplets_from_parquet(
        args.test_parquet,
        drug_to_idx=drug_to_idx,
        target_to_idx=target_to_idx,
        disease_to_idx=disease_to_idx,
    )
    if args.max_test_rows is not None and args.max_test_rows < int(encoded.labels.shape[0]):
        subset = np.arange(args.max_test_rows, dtype=np.int64)
        encoded = triplet.EncodedTriplets(
            drug_idx=encoded.drug_idx[subset],
            target_idx=encoded.target_idx[subset],
            disease_idx=encoded.disease_idx[subset],
            labels=encoded.labels[subset],
            drug_ids=encoded.drug_ids[subset],
            target_ids=encoded.target_ids[subset],
            disease_ids=encoded.disease_ids[subset],
        )
    unk_counts = {
        "drug": int((encoded.drug_idx == 0).sum()),
        "target": int((encoded.target_idx == 0).sum()),
        "disease": int((encoded.disease_idx == 0).sum()),
    }

    dataset = triplet.TripletDataset(
        encoded.drug_idx,
        encoded.target_idx,
        encoded.disease_idx,
        encoded.labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = triplet.TripletLSTMClassifier(
        drug_embedding_table=checkpoint["drug_embedding_table"].cpu().numpy(),
        target_embedding_table=checkpoint["target_embedding_table"].cpu().numpy(),
        disease_embedding_table=checkpoint["disease_embedding_table"].cpu().numpy(),
        lstm_hidden_dim=int(checkpoint["lstm_hidden_dim"]),
        mlp_hidden_dim=int(checkpoint["mlp_hidden_dim"]),
        mlp_layers=int(checkpoint["mlp_layers"]),
        dropout=float(checkpoint["dropout"]),
        freeze_embeddings=bool(checkpoint["freeze_embeddings"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    probs, labels = triplet.run_inference(model, loader, device)
    pred_df = pd.DataFrame(
        {
            "drug_id_norm": encoded.drug_ids,
            "target_id_norm": encoded.target_ids,
            "disease_id_norm": encoded.disease_ids,
            "label": labels.astype(np.int64),
            "score": probs.astype(np.float32),
        }
    )
    pred_df["pred"] = (pred_df["score"].to_numpy() >= args.threshold).astype(np.int64)
    triplet_pred_path = os.path.join(args.output_dir, "test_triplet_predictions.parquet")
    pred_df.to_parquet(triplet_pred_path, index=False)

    triplet_metrics = triplet.compute_classification_metrics(
        labels=labels, probs=probs, threshold=args.threshold
    )
    triplet_metrics["num_rows"] = int(len(labels))
    triplet_metrics["threshold"] = float(args.threshold)
    triplet_metrics["unk_counts"] = unk_counts
    triplet_metrics["enrichment_factors"] = triplet.compute_enrichment_factors(
        labels=labels, scores=probs, fracs=ef_fracs
    )

    pair_df = triplet.aggregate_pair_predictions(pred_df, pair_cols=("drug_id_norm", "disease_id_norm"))
    pair_df["pred"] = (pair_df["score"].to_numpy() >= args.threshold).astype(np.int64)
    pair_pred_path = os.path.join(args.output_dir, "test_pair_predictions.parquet")
    pair_df.to_parquet(pair_pred_path, index=False)

    pair_labels = pair_df["label"].to_numpy(dtype=np.int64)
    pair_scores = pair_df["score"].to_numpy(dtype=np.float32)
    pair_metrics = triplet.compute_classification_metrics(
        labels=pair_labels, probs=pair_scores, threshold=args.threshold
    )
    pair_metrics["num_pairs"] = int(pair_df.shape[0])
    pair_metrics["threshold"] = float(args.threshold)
    pair_metrics["pair_key"] = "drug_id_norm+disease_id_norm"
    pair_metrics["enrichment_factors"] = triplet.compute_enrichment_factors(
        labels=pair_labels, scores=pair_scores, fracs=ef_fracs
    )

    triplet.save_json(os.path.join(args.output_dir, "metrics_triplet.json"), triplet_metrics)
    triplet.save_json(os.path.join(args.output_dir, "metrics_pair.json"), pair_metrics)

    print("Triplet metrics:", triplet_metrics)
    print("Pair metrics:", pair_metrics)
    print(f"Saved triplet predictions: {triplet_pred_path}")
    print(f"Saved pair predictions: {pair_pred_path}")


if __name__ == "__main__":
    main()
