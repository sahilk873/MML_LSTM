import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polypharmacy_kedro import data as data_lib  # noqa: E402
from polypharmacy_kedro import model as model_lib  # noqa: E402
from polypharmacy_kedro import utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate polypharmacy predictor.")
    parser.add_argument("--checkpoint", default="data/06_models/best_model.pt")
    parser.add_argument("--drug-embeddings", default="data/06_models/drug_embeddings.pkl")
    parser.add_argument(
        "--disease-embeddings", default="data/06_models/disease_embeddings.pkl"
    )
    parser.add_argument("--run-dataset", default="data/04_feature/filtered_dataset_run.csv")
    parser.add_argument("--val-splits", default="data/05_model_input/val_idx.pkl")
    parser.add_argument("--test-splits", default="data/05_model_input/test_idx.pkl")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def evaluate_split(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
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


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.run_dataset):
        raise FileNotFoundError(
            "Run dataset not found. Run `kedro run` to generate artifacts first."
        )

    with open(args.checkpoint, "rb") as handle:
        checkpoint = pickle.load(handle)
    drug_embeddings = None
    disease_embeddings = None

    if args.drug_embeddings.endswith(".pkl"):
        with open(args.drug_embeddings, "rb") as handle:
            drug_embeddings = pickle.load(handle)
    if args.disease_embeddings.endswith(".pkl"):
        with open(args.disease_embeddings, "rb") as handle:
            disease_embeddings = pickle.load(handle)

    if drug_embeddings is None or disease_embeddings is None:
        raise ValueError("Unable to load embeddings from the provided paths.")

    run_df = pd.read_csv(args.run_dataset)
    run_df["drug_set"] = run_df["drug_set"].apply(data_lib.parse_list_column)

    examples = data_lib.dataframe_to_examples(run_df)
    drug_to_idx, disease_to_idx = data_lib.build_mappings(examples)
    drug_seqs, disease_idxs, labels = data_lib.encode_examples(
        examples, drug_to_idx, disease_to_idx
    )

    with open(args.val_splits, "rb") as handle:
        val_idx = pickle.load(handle)
    with open(args.test_splits, "rb") as handle:
        test_idx = pickle.load(handle)

    dataset = data_lib.PolypharmacyDataset(drug_seqs, disease_idxs, labels)
    collate = lambda batch: data_lib.collate_batch(batch, pad_idx=0)
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    val_metrics = evaluate_split(model, val_loader, device)
    test_metrics = evaluate_split(model, test_loader, device)

    def format_metrics(name: str, metrics: dict) -> str:
        return (
            f"{name:<11} | auc={metrics['roc_auc']:.4f} | acc={metrics['accuracy']:.4f} "
            f"| sens={metrics['sensitivity']:.4f} | spec={metrics['specificity']:.4f} "
            f"| f1={metrics['f1']:.4f} | confusion={metrics['confusion']}"
        )

    print(format_metrics("Validation", val_metrics))
    print(format_metrics("Test", test_metrics))


if __name__ == "__main__":
    main()
