#!/usr/bin/env python3
import argparse
import glob
import os
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate precomputed KG embeddings from nodes_with_embeddings/*.parquet "
            "into a single node_id + embedding matrix file."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="nodes_with_embeddings",
        help="Directory containing nodes_with_embeddings parquet parts.",
    )
    parser.add_argument(
        "--embedding-type",
        choices=["pca", "topological", "concat"],
        default="topological",
        help="Embedding column to extract (concat = pca + topological).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("artifacts", "precomputed_embeddings"),
        help="Directory to write embeddings.npy and node_ids.npy.",
    )
    parser.add_argument(
        "--id-prefixes",
        nargs="*",
        default=None,
        help="Optional list of ID prefixes to keep (e.g., CHEBI MONDO UNII).",
    )
    parser.add_argument(
        "--required-ids",
        default=None,
        help=(
            "Optional path to a file containing node IDs to keep, one per line. "
            "If set, only these IDs are retained."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows to export after filtering.",
    )
    return parser.parse_args()


def load_required_ids(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def iter_parts(paths: Iterable[str], columns: List[str]) -> Iterable[pd.DataFrame]:
    for path in paths:
        yield pd.read_parquet(path, columns=columns)


def filter_ids(
    df: pd.DataFrame,
    required_ids: Optional[Set[str]],
    id_prefixes: Optional[Set[str]],
) -> pd.DataFrame:
    df = df[df["id"].notna()]
    if required_ids is not None:
        df = df[df["id"].isin(required_ids)]
    if id_prefixes:
        df = df[df["id"].str.split(":", n=1).str[0].isin(id_prefixes)]
    return df


def select_embedding(
    df: pd.DataFrame, embedding_type: str
) -> Tuple[np.ndarray, List[str]]:
    if embedding_type == "pca":
        embeddings = np.vstack(df["pca_embedding"].to_list()).astype(np.float32)
    elif embedding_type == "topological":
        embeddings = np.vstack(df["topological_embedding"].to_list()).astype(np.float32)
    else:
        pca = np.vstack(df["pca_embedding"].to_list()).astype(np.float32)
        topo = np.vstack(df["topological_embedding"].to_list()).astype(np.float32)
        embeddings = np.concatenate([pca, topo], axis=1)
    return embeddings, df["id"].tolist()


def count_rows(
    paths: List[str],
    required_ids: Optional[Set[str]],
    id_prefixes: Optional[Set[str]],
    embedding_type: str,
) -> int:
    total = 0
    cols = ["id"]
    if embedding_type in ("pca", "concat"):
        cols.append("pca_embedding")
    if embedding_type in ("topological", "concat"):
        cols.append("topological_embedding")
    for df in iter_parts(paths, columns=cols):
        df = filter_ids(df, required_ids, id_prefixes)
        if embedding_type == "pca":
            df = df[df["pca_embedding"].notna()]
        elif embedding_type == "topological":
            df = df[df["topological_embedding"].notna()]
        else:
            df = df[df["pca_embedding"].notna() & df["topological_embedding"].notna()]
        total += len(df)
    return total


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not paths:
        raise SystemExit(f"No parquet parts found in {args.input_dir}")

    required_ids = load_required_ids(args.required_ids)
    id_prefixes = set(args.id_prefixes) if args.id_prefixes else None

    total_rows = count_rows(paths, required_ids, id_prefixes, args.embedding_type)
    if args.max_rows is not None:
        total_rows = min(total_rows, args.max_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    ids_path = os.path.join(args.output_dir, "node_ids.npy")
    emb_path = os.path.join(args.output_dir, "embeddings.npy")

    # Determine embedding dimension using the first available row.
    sample = pd.read_parquet(
        paths[0], columns=["pca_embedding", "topological_embedding"]
    ).iloc[0]
    if args.embedding_type == "pca":
        emb_dim = len(sample["pca_embedding"])
    elif args.embedding_type == "topological":
        emb_dim = len(sample["topological_embedding"])
    else:
        emb_dim = len(sample["pca_embedding"]) + len(sample["topological_embedding"])

    embeddings = np.lib.format.open_memmap(
        emb_path, mode="w+", dtype="float32", shape=(total_rows, emb_dim)
    )
    node_ids: List[str] = []

    cursor = 0
    for df in iter_parts(
        paths, columns=["id", "pca_embedding", "topological_embedding"]
    ):
        df = filter_ids(df, required_ids, id_prefixes)
        if args.embedding_type == "pca":
            df = df[df["pca_embedding"].notna()]
        elif args.embedding_type == "topological":
            df = df[df["topological_embedding"].notna()]
        else:
            df = df[df["pca_embedding"].notna() & df["topological_embedding"].notna()]
        if df.empty:
            continue
        batch_embeddings, batch_ids = select_embedding(df, args.embedding_type)
        if args.max_rows is not None and cursor + len(batch_ids) > total_rows:
            keep = total_rows - cursor
            batch_embeddings = batch_embeddings[:keep]
            batch_ids = batch_ids[:keep]
        embeddings[cursor : cursor + len(batch_ids)] = batch_embeddings
        node_ids.extend(batch_ids)
        cursor += len(batch_ids)
        if args.max_rows is not None and cursor >= total_rows:
            break

    if cursor != total_rows:
        raise RuntimeError(
            f"Row count mismatch: expected {total_rows}, wrote {cursor}. "
            "Rerun with a smaller filter or inspect missing embeddings."
        )

    np.save(ids_path, np.array(node_ids, dtype=object))

    print(
        f"Wrote {total_rows} embeddings (dim={emb_dim}) to {emb_path} "
        f"and node IDs to {ids_path}"
    )


if __name__ == "__main__":
    main()
