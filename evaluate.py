import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from polypharmacy import data as data_lib
from polypharmacy import kg as kg_lib
from polypharmacy import model as model_lib
from polypharmacy import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate polypharmacy predictor.")
    parser.add_argument("--config", default=None, help="Optional JSON config override.")
    parser.add_argument("--indications", default="indications_norm.csv")
    parser.add_argument("--contraindications", default="contraindications_norm.csv")
    parser.add_argument(
        "--single-therapy-indications",
        default=None,
        help="Optional RENCI single-therapy indications CSV.",
    )
    parser.add_argument(
        "--single-therapy-contraindications",
        default=None,
        help="Optional RENCI single-therapy contraindications CSV.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kg", default="kg_edges.parquet")
    parser.add_argument("--edge-src-col", default=None)
    parser.add_argument("--edge-dst-col", default=None)
    parser.add_argument(
        "--disease-token-position",
        choices=["first", "last", "none"],
        default=None,
        help="Override checkpoint: inject disease embedding as a token in the LSTM sequence.",
    )
    parser.add_argument(
        "--concat-disease-after-lstm",
        choices=["true", "false"],
        default=None,
        help="Override checkpoint: concat disease embedding after LSTM.",
    )
    return parser.parse_args()


def evaluate_split(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict:
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
    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    drug_embeddings = np.load(os.path.join(args.output_dir, "drug_embeddings.npy"))
    disease_embeddings = np.load(os.path.join(args.output_dir, "disease_embeddings.npy"))

    run_path = os.path.join(args.output_dir, "filtered_dataset_run.csv")
    filtered_path = os.path.join(args.output_dir, "filtered_dataset.csv")
    if os.path.exists(run_path):
        filtered_df = pd.read_csv(run_path)
        filtered_df["drug_set"] = filtered_df["drug_set"].apply(data_lib.parse_list_column)
    elif os.path.exists(filtered_path):
        filtered_df = pd.read_csv(filtered_path)
        filtered_df["drug_set"] = filtered_df["drug_set"].apply(data_lib.parse_list_column)
    else:
        deduped_df, _ = data_lib.load_deduped_dataframe(
            args.indications,
            args.contraindications,
            single_therapy_indications_path=args.single_therapy_indications,
            single_therapy_contraindications_path=args.single_therapy_contraindications,
        )
        edges = kg_lib.load_edges(args.kg, src_col=args.edge_src_col, dst_col=args.edge_dst_col)
        kg_nodes = kg_lib.extract_kg_nodes(edges)
        filtered_df, _, _ = data_lib.filter_by_kg_coverage(deduped_df, kg_nodes)

    examples = data_lib.dataframe_to_examples(filtered_df)
    drug_vocab_path = os.path.join(args.output_dir, "drug_vocab.json")
    disease_vocab_path = os.path.join(args.output_dir, "disease_vocab.json")

    if os.path.exists(drug_vocab_path) and os.path.exists(disease_vocab_path):
        drug_ids = utils.load_json(drug_vocab_path)["ids"]
        disease_ids = utils.load_json(disease_vocab_path)["ids"]
        drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids) if idx != 0}
        disease_to_idx = {disease_id: idx for idx, disease_id in enumerate(disease_ids)}
    else:
        drug_to_idx, disease_to_idx = data_lib.build_mappings(examples)
    drug_seqs, disease_idxs, labels = data_lib.encode_examples(
        examples, drug_to_idx, disease_to_idx
    )

    splits = np.load(os.path.join(args.output_dir, "splits.npz"))
    if splits.get("num_examples", None) != len(labels):
        raise ValueError(
            "Split file does not match current dataset size. "
            "Re-run training to regenerate splits."
        )
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

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
        disease_token_position=(
            None
            if args.disease_token_position == "none"
            else args.disease_token_position
        )
        if args.disease_token_position is not None
        else checkpoint.get("disease_token_position"),
        concat_disease_after_lstm=(
            args.concat_disease_after_lstm == "true"
            if args.concat_disease_after_lstm is not None
            else checkpoint.get("concat_disease_after_lstm", True)
        ),
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
