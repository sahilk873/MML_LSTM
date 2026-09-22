import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from polypharmacy import triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train triplet LSTM baseline on train_kushal.parquet.")
    parser.add_argument("--train-parquet", default="train_kushal.parquet")
    parser.add_argument("--output-dir", default="artifacts_triplet_lstm")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_tables(train_path: str, seed: int) -> Dict[str, object]:
    drug_map = triplet.build_embedding_map(train_path, "drug_id_norm", "drug_embedding")
    target_map = triplet.build_embedding_map(train_path, "target_id_norm", "target_embedding")
    disease_map = triplet.build_embedding_map(train_path, "disease_id_norm", "disease_embedding")

    drug_vocab, drug_table = triplet.build_vocab_and_embedding_table(drug_map, seed=seed)
    target_vocab, target_table = triplet.build_vocab_and_embedding_table(target_map, seed=seed + 1)
    disease_vocab, disease_table = triplet.build_vocab_and_embedding_table(disease_map, seed=seed + 2)
    return {
        "drug_vocab": drug_vocab,
        "target_vocab": target_vocab,
        "disease_vocab": disease_vocab,
        "drug_table": drug_table,
        "target_table": target_table,
        "disease_table": disease_table,
    }


def main() -> None:
    args = parse_args()
    triplet.set_all_seeds(args.seed)
    triplet.validate_parquet_schema(args.train_parquet)
    ensure_dir(args.output_dir)

    tables = build_tables(args.train_parquet, seed=args.seed)
    drug_to_idx = triplet.ids_to_index_map(tables["drug_vocab"])
    target_to_idx = triplet.ids_to_index_map(tables["target_vocab"])
    disease_to_idx = triplet.ids_to_index_map(tables["disease_vocab"])

    encoded = triplet.encode_triplets_from_parquet(
        args.train_parquet, drug_to_idx, target_to_idx, disease_to_idx
    )
    if args.max_train_rows is not None and args.max_train_rows < int(encoded.labels.shape[0]):
        all_idx = np.arange(int(encoded.labels.shape[0]))
        sampled_idx, _ = train_test_split(
            all_idx,
            train_size=args.max_train_rows,
            random_state=args.seed,
            shuffle=True,
            stratify=encoded.labels,
        )
        sampled_idx = np.sort(sampled_idx)
        encoded = triplet.EncodedTriplets(
            drug_idx=encoded.drug_idx[sampled_idx],
            target_idx=encoded.target_idx[sampled_idx],
            disease_idx=encoded.disease_idx[sampled_idx],
            labels=encoded.labels[sampled_idx],
            drug_ids=encoded.drug_ids[sampled_idx],
            target_ids=encoded.target_ids[sampled_idx],
            disease_ids=encoded.disease_ids[sampled_idx],
        )

    num_examples = int(encoded.labels.shape[0])
    all_idx = np.arange(num_examples)
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=args.val_frac,
        random_state=args.seed,
        shuffle=True,
        stratify=encoded.labels,
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "train_val_split_indices.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
        num_examples=num_examples,
        seed=args.seed,
    )

    dataset = triplet.TripletDataset(
        encoded.drug_idx,
        encoded.target_idx,
        encoded.disease_idx,
        encoded.labels,
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = triplet.TripletLSTMClassifier(
        drug_embedding_table=tables["drug_table"],
        target_embedding_table=tables["target_table"],
        disease_embedding_table=tables["disease_table"],
        lstm_hidden_dim=args.lstm_hidden_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        freeze_embeddings=args.freeze_embeddings,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    log_path = os.path.join(args.output_dir, "train_log.csv")
    best_path = os.path.join(args.output_dir, "best_model.pt")
    best_auc = float("-inf")
    patience_count = 0

    with open(log_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_roc_auc",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_balanced_accuracy",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for drug_idx, target_idx, disease_idx, labels in train_loader:
                drug_idx = drug_idx.to(device)
                target_idx = target_idx.to(device)
                disease_idx = disease_idx.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(drug_idx, target_idx, disease_idx)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item())

            avg_train_loss = loss_sum / max(1, len(train_loader))
            val_probs, val_labels = triplet.run_inference(model, val_loader, device)
            val_metrics = triplet.compute_classification_metrics(val_labels, val_probs, threshold=0.5)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                }
            )
            handle.flush()

            print(
                f"Epoch {epoch:02d} | loss={avg_train_loss:.4f} | "
                f"val_auc={val_metrics['roc_auc']:.4f} | "
                f"val_prec={val_metrics['precision']:.4f} | "
                f"val_rec={val_metrics['recall']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
            )

            current_auc = val_metrics["roc_auc"]
            if not np.isnan(current_auc) and current_auc > best_auc:
                best_auc = current_auc
                patience_count = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "seed": args.seed,
                        "lstm_hidden_dim": args.lstm_hidden_dim,
                        "mlp_hidden_dim": args.mlp_hidden_dim,
                        "mlp_layers": args.mlp_layers,
                        "dropout": args.dropout,
                        "freeze_embeddings": args.freeze_embeddings,
                        "drug_vocab": tables["drug_vocab"],
                        "target_vocab": tables["target_vocab"],
                        "disease_vocab": tables["disease_vocab"],
                        "drug_embedding_table": torch.tensor(tables["drug_table"]),
                        "target_embedding_table": torch.tensor(tables["target_table"]),
                        "disease_embedding_table": torch.tensor(tables["disease_table"]),
                        "best_val_metrics": val_metrics,
                    },
                    best_path,
                )
            else:
                patience_count += 1
                if patience_count >= args.early_stop_patience:
                    print(
                        "Early stopping triggered after "
                        f"{args.early_stop_patience} epochs without val ROC-AUC improvement."
                    )
                    break

    config_payload: Dict[str, object] = {
        "train_parquet": args.train_parquet,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_layers": args.mlp_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "early_stop_patience": args.early_stop_patience,
        "freeze_embeddings": args.freeze_embeddings,
        "num_examples": num_examples,
        "train_size": int(train_idx.shape[0]),
        "val_size": int(val_idx.shape[0]),
    }
    triplet.save_json(os.path.join(args.output_dir, "config.json"), config_payload)

    checkpoint = torch.load(best_path, map_location="cpu")
    triplet.save_json(
        os.path.join(args.output_dir, "val_metrics_best.json"),
        checkpoint["best_val_metrics"],
    )
    print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
