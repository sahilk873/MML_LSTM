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


def load_filtered_df(args: argparse.Namespace) -> pd.DataFrame:
    run_path = os.path.join(args.output_dir, "filtered_dataset_run.csv")
    filtered_path = os.path.join(args.output_dir, "filtered_dataset.csv")
    if os.path.exists(run_path):
        filtered_df = pd.read_csv(run_path)
        filtered_df["drug_set"] = filtered_df["drug_set"].apply(data_lib.parse_list_column)
        return filtered_df
    if os.path.exists(filtered_path):
        filtered_df = pd.read_csv(filtered_path)
        filtered_df["drug_set"] = filtered_df["drug_set"].apply(data_lib.parse_list_column)
        return filtered_df
    deduped_df, _ = data_lib.load_deduped_dataframe(
        args.indications,
        args.contraindications,
        single_therapy_indications_path=args.single_therapy_indications,
        single_therapy_contraindications_path=args.single_therapy_contraindications,
    )
    edges = kg_lib.load_edges(args.kg, src_col=args.edge_src_col, dst_col=args.edge_dst_col)
    kg_nodes = kg_lib.extract_kg_nodes(edges)
    filtered_df, _, _ = data_lib.filter_by_kg_coverage(deduped_df, kg_nodes)
    return filtered_df


def resolve_embedding_path(output_dir: str, filename: str) -> str:
    primary = os.path.join(output_dir, filename)
    if os.path.exists(primary):
        return primary
    return primary


def aggregate_metrics(metrics_list: list) -> dict:
    metric_keys = ["roc_auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {}
    for key in metric_keys:
        values = np.array([float(m[key]) for m in metrics_list], dtype=float)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan"), "sem": float("nan")}
            continue
        mean = float(np.mean(valid))
        if len(valid) > 1:
            std = float(np.std(valid, ddof=1))
            sem = float(std / np.sqrt(len(valid)))
        else:
            std = 0.0
            sem = 0.0
        summary[key] = {"mean": mean, "std": std, "sem": sem}
    return summary


def format_mean_sem(value: dict) -> str:
    return f"{value['mean']:.4f}±{value['sem']:.4f}"


def format_summary(label: str, summary: dict) -> str:
    return (
        f"{label:<11} | auc={format_mean_sem(summary['roc_auc'])} | "
        f"acc={format_mean_sem(summary['accuracy'])} | "
        f"sens={format_mean_sem(summary['sensitivity'])} | "
        f"spec={format_mean_sem(summary['specificity'])} | "
        f"f1={format_mean_sem(summary['f1'])}"
    )


def evaluate_run(
    args: argparse.Namespace,
    run_dir: str,
    filtered_df: pd.DataFrame,
    drug_embeddings: np.ndarray,
    disease_embeddings: np.ndarray,
    drug_to_idx: dict,
    disease_to_idx: dict,
    checkpoint_path: str,
) -> tuple:
    examples = data_lib.dataframe_to_examples(filtered_df)
    drug_seqs, disease_idxs, labels = data_lib.encode_examples(
        examples, drug_to_idx, disease_to_idx
    )

    splits = np.load(os.path.join(run_dir, "splits.npz"))
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

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
    return val_metrics, test_metrics


def main() -> None:
    args = parse_args()
    filtered_df = load_filtered_df(args)

    drug_embeddings_path = resolve_embedding_path(args.output_dir, "drug_embeddings.npy")
    disease_embeddings_path = resolve_embedding_path(args.output_dir, "disease_embeddings.npy")
    drug_embeddings = np.load(drug_embeddings_path)
    disease_embeddings = np.load(disease_embeddings_path)

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

    run_dirs = sorted(
        [
            os.path.join(args.output_dir, name)
            for name in os.listdir(args.output_dir)
            if name.startswith("run_")
            and os.path.isdir(os.path.join(args.output_dir, name))
        ]
    )

    def format_metrics(name: str, metrics: dict) -> str:
        return (
            f"{name:<11} | auc={metrics['roc_auc']:.4f} | acc={metrics['accuracy']:.4f} "
            f"| sens={metrics['sensitivity']:.4f} | spec={metrics['specificity']:.4f} "
            f"| f1={metrics['f1']:.4f} | confusion={metrics['confusion']}"
        )

    val_metrics_all = []
    test_metrics_all = []

    if run_dirs:
        for run_dir in run_dirs:
            default_checkpoint = os.path.join(run_dir, "best_model.pt")
            if args.checkpoint:
                if (
                    args.checkpoint == os.path.join(args.output_dir, "best_model.pt")
                    and os.path.exists(default_checkpoint)
                ):
                    checkpoint_path = default_checkpoint
                else:
                    checkpoint_path = args.checkpoint
            else:
                checkpoint_path = default_checkpoint

            val_metrics, test_metrics = evaluate_run(
                args,
                run_dir,
                filtered_df,
                drug_embeddings,
                disease_embeddings,
                drug_to_idx,
                disease_to_idx,
                checkpoint_path,
            )
            run_name = os.path.basename(run_dir)
            print(f"{run_name} | {format_metrics('Validation', val_metrics)}")
            print(f"{run_name} | {format_metrics('Test', test_metrics)}")
            val_metrics_all.append(val_metrics)
            test_metrics_all.append(test_metrics)

        if val_metrics_all and test_metrics_all:
            val_summary = aggregate_metrics(val_metrics_all)
            test_summary = aggregate_metrics(test_metrics_all)
            print(format_summary("Validation", val_summary))
            print(format_summary("Test", test_summary))
        return

    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, "best_model.pt")
    val_metrics, test_metrics = evaluate_run(
        args,
        args.output_dir,
        filtered_df,
        drug_embeddings,
        disease_embeddings,
        drug_to_idx,
        disease_to_idx,
        checkpoint_path,
    )
    print(format_metrics("Validation", val_metrics))
    print(format_metrics("Test", test_metrics))


if __name__ == "__main__":
    main()
