#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import subprocess
from typing import Dict, Iterable, List


def parse_list(arg: str, cast=int) -> List:
    return [cast(x) for x in arg.split(",") if x.strip()]


def load_base_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_config(path: str, config: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def run_command(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep experiment.py hyperparameters.")
    parser.add_argument("--base-config", default="config_override.json")
    parser.add_argument("--output-root", default="artifacts/sweeps")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--kg", default="kg_edges.parquet")
    parser.add_argument("--kg-workers", type=int, default=4)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=16)
    parser.add_argument("--single-therapy-indications", default=None)
    parser.add_argument("--single-therapy-contraindications", default=None)
    parser.add_argument(
        "--embedding-sources",
        default="precomputed,kg",
        help="Comma-separated: precomputed,kg",
    )
    parser.add_argument("--precomputed-embeddings", default=None)
    parser.add_argument("--precomputed-embedding-ids", default=None)
    parser.add_argument("--lstm-hidden-dims", default="128,256")
    parser.add_argument("--mlp-hidden-dims", default="128,256")
    parser.add_argument("--mlp-layers", default="2,3")
    parser.add_argument(
        "--disease-token-positions",
        default="none",
        help="Comma-separated: none,first,last",
    )
    parser.add_argument(
        "--concat-disease-after-lstm",
        default="true",
        help="Comma-separated: true,false",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs if output directory already exists.",
    )
    args = parser.parse_args()

    base_config = load_base_config(args.base_config)
    embedding_sources = [s.strip() for s in args.embedding_sources.split(",") if s.strip()]
    lstm_dims = parse_list(args.lstm_hidden_dims)
    mlp_dims = parse_list(args.mlp_hidden_dims)
    mlp_layers = parse_list(args.mlp_layers)
    disease_positions = [s.strip() for s in args.disease_token_positions.split(",") if s.strip()]
    concat_after = [s.strip() for s in args.concat_disease_after_lstm.split(",") if s.strip()]

    if "precomputed" in embedding_sources:
        if not args.precomputed_embeddings:
            raise ValueError("precomputed embeddings require --precomputed-embeddings")
        if args.precomputed_embeddings.endswith(".npy") and not args.precomputed_embedding_ids:
            raise ValueError(
                "precomputed .npy embeddings require --precomputed-embedding-ids"
            )

    os.makedirs(args.output_root, exist_ok=True)

    sweep = itertools.product(
        embedding_sources,
        lstm_dims,
        mlp_dims,
        mlp_layers,
        disease_positions,
        concat_after,
    )
    for (
        source,
        lstm_dim,
        mlp_dim,
        mlp_layers_count,
        disease_pos,
        concat_flag,
    ) in sweep:
        if disease_pos == "none" and concat_flag != "true":
            continue
        run_name = (
            f"{source}_lstm{lstm_dim}_mlp{mlp_dim}_layers{mlp_layers_count}"
            f"_disease{disease_pos}_concat{concat_flag}"
        )
        output_dir = os.path.join(args.output_root, run_name)
        if args.skip_existing and os.path.exists(output_dir):
            print(f"Skipping existing run: {output_dir}")
            continue

        run_config = dict(base_config)
        run_config.update(
            {
                "lstm_hidden_dim": lstm_dim,
                "mlp_hidden_dim": mlp_dim,
                "mlp_layers": mlp_layers_count,
                "disease_token_position": None if disease_pos == "none" else disease_pos,
                "concat_disease_after_lstm": concat_flag == "true",
            }
        )
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "config.json")
        write_config(config_path, run_config)

        cmd = [
            "python3",
            "experiment.py",
            "--config",
            config_path,
            "--kg",
            args.kg,
            "--kg-workers",
            str(args.kg_workers),
            "--output-dir",
            output_dir,
            "--rf-estimators",
            str(args.rf_estimators),
            "--rf-max-depth",
            str(args.rf_max_depth),
            "--test-frac",
            str(args.test_frac),
            "--seed",
            str(args.seed),
        ]
        if args.single_therapy_indications:
            cmd.extend(["--single-therapy-indications", args.single_therapy_indications])
        if args.single_therapy_contraindications:
            cmd.extend(
                [
                    "--single-therapy-contraindications",
                    args.single_therapy_contraindications,
                ]
            )

        if source == "precomputed":
            cmd.extend(
                [
                    "--kg-embeddings",
                    args.precomputed_embeddings,
                    "--kg-embedding-ids",
                    args.precomputed_embedding_ids,
                ]
            )
        else:
            cmd.extend(["--kg-hop-expansion", "0"])

        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
        else:
            run_command(cmd)


if __name__ == "__main__":
    main()
