#!/usr/bin/env python3
import argparse
import ast
import glob
import os
import shutil
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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
    parser.add_argument(
        "--write-equivalent-id-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write alias index parquet mapping equivalent identifiers to canonical node IDs.",
    )
    parser.add_argument(
        "--equivalent-id-index-name",
        default="equivalent_id_to_node_id.parquet",
        help="Filename for alias index parquet written under output-dir.",
    )
    parser.add_argument(
        "--only-equivalent-id-index",
        action="store_true",
        help="Skip embeddings/node_ids export and only write alias index parquet.",
    )
    parser.add_argument(
        "--extra-equivalent-id-output-dirs",
        nargs="*",
        default=None,
        help=(
            "Optional additional output directories that should receive a copy of "
            "the alias index parquet."
        ),
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


def _is_invalid_alias(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return True
    lower = text.lower()
    if lower in {"nan", "none"}:
        return True
    if lower.startswith("error") or text.startswith("['Error"):
        return True
    return False


def _parse_equivalent_identifiers(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, list):
        values = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                values = parsed
            elif isinstance(parsed, str):
                values = [parsed]
            else:
                values = [text]
        except (ValueError, SyntaxError):
            values = [token.strip().strip("'") for token in text.strip("[]").split(",")]
    else:
        values = [value]

    cleaned: List[str] = []
    for token in values:
        alias = str(token).strip()
        if _is_invalid_alias(alias):
            continue
        cleaned.append(alias)
    return cleaned


def iter_equivalent_id_rows(
    paths: List[str],
    required_ids: Optional[Set[str]],
    id_prefixes: Optional[Set[str]],
) -> Iterable[pd.DataFrame]:
    for df in iter_parts(paths, columns=["id", "equivalent_identifiers"]):
        df = filter_ids(df, required_ids, id_prefixes)
        if df.empty:
            continue
        aliases: List[str] = []
        node_ids: List[str] = []
        sources: List[str] = []
        for row in df.itertuples(index=False):
            node_id = str(row.id).strip()
            if _is_invalid_alias(node_id):
                continue
            local_aliases = {node_id}
            for alias in _parse_equivalent_identifiers(row.equivalent_identifiers):
                local_aliases.add(alias)
            for alias in sorted(local_aliases):
                aliases.append(alias)
                node_ids.append(node_id)
                sources.append("id" if alias == node_id else "equivalent_identifiers")
        chunk = pd.DataFrame(
            {
                "alias_id": aliases,
                "node_id": node_ids,
                "match_source": sources,
            }
        )
        if chunk.empty:
            continue
        # Remove duplicates emitted within this chunk before writing.
        yield chunk.drop_duplicates(subset=["alias_id", "node_id"])


def write_equivalent_id_index(
    paths: List[str],
    output_path: str,
    required_ids: Optional[Set[str]],
    id_prefixes: Optional[Set[str]],
) -> int:
    schema = pa.schema(
        [
            ("alias_id", pa.string()),
            ("node_id", pa.string()),
            ("match_source", pa.string()),
        ]
    )
    writer = pq.ParquetWriter(output_path, schema=schema)
    row_count = 0
    seen_pairs: Set[Tuple[str, str]] = set()
    try:
        for chunk in iter_equivalent_id_rows(paths, required_ids, id_prefixes):
            keep_mask = []
            for row in chunk.itertuples(index=False):
                key = (str(row.alias_id), str(row.node_id))
                if key in seen_pairs:
                    keep_mask.append(False)
                    continue
                seen_pairs.add(key)
                keep_mask.append(True)
            deduped = chunk.loc[keep_mask]
            if deduped.empty:
                continue
            table = pa.Table.from_pandas(deduped, schema=schema, preserve_index=False)
            writer.write_table(table)
            row_count += len(deduped)
    finally:
        writer.close()
    return row_count


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

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.only_equivalent_id_index:
        total_rows = count_rows(paths, required_ids, id_prefixes, args.embedding_type)
        if args.max_rows is not None:
            total_rows = min(total_rows, args.max_rows)

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
    equiv_rows = 0
    equiv_path = os.path.join(args.output_dir, args.equivalent_id_index_name)
    if args.write_equivalent_id_index:
        equiv_rows = write_equivalent_id_index(
            paths=paths,
            output_path=equiv_path,
            required_ids=required_ids,
            id_prefixes=id_prefixes,
        )
        for extra_dir in args.extra_equivalent_id_output_dirs or []:
            os.makedirs(extra_dir, exist_ok=True)
            extra_path = os.path.join(extra_dir, args.equivalent_id_index_name)
            shutil.copy2(equiv_path, extra_path)
            print(f"Copied alias index to {extra_path}")

    if not args.only_equivalent_id_index:
        print(
            f"Wrote {total_rows} embeddings (dim={emb_dim}) to {emb_path} "
            f"and node IDs to {ids_path}"
        )
    if args.write_equivalent_id_index:
        print(f"Wrote {equiv_rows} alias mappings to {equiv_path}")


if __name__ == "__main__":
    main()
