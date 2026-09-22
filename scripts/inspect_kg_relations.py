import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pyarrow.parquet as pq


def iter_standard_column(path: str, column: str) -> Iterator[str]:
    """Yield values for an existing KG edge column (subject, predicate, object)."""
    parquet_file = pq.ParquetFile(path)
    for rg_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(rg_index, columns=[column])
        yield from table[column].to_pylist()


def iter_category_column(
    path: str,
    side: str,
    node_categories: dict[str, str],
) -> Iterator[str]:
    """Yield the category for each subject/object node identified in `side`."""
    parquet_file = pq.ParquetFile(path)
    for rg_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(rg_index, columns=[side])
        for node_id in table[side].to_pylist():
            yield node_categories.get(node_id, "UNKNOWN")


def load_node_categories(path: Path) -> dict[str, str]:
    """Load node category labels so we can summarize subject/object categories."""
    categories: dict[str, str] = {}
    parquet_paths = sorted(path.glob("*.parquet"))
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(str(parquet_path))
        for rg_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(rg_index, columns=["id", "category"])
            ids = table["id"].to_pylist()
            cats = table["category"].to_pylist()
            for node_id, category in zip(ids, cats):
                if node_id is None:
                    continue
                categories[node_id] = category or "UNKNOWN"
    return categories


def summarize_column(
    path: str,
    column: str,
    top_k: int,
    min_count: int,
    search_terms: List[str],
    node_categories: Optional[dict[str, str]] = None,
) -> None:
    """Print counts and optional search matches for a single column."""
    if column in {"subject", "predicate", "object"}:
        iterator = iter_standard_column(path, column)
    elif column == "subject_category":
        if node_categories is None:
            raise ValueError(
                "subject_category requested but node categories were not loaded."
            )
        iterator = iter_category_column(path, "subject", node_categories)
    elif column == "object_category":
        if node_categories is None:
            raise ValueError(
                "object_category requested but node categories were not loaded."
            )
        iterator = iter_category_column(path, "object", node_categories)
    else:
        raise ValueError(f"Unsupported column {column!r}")
    counter = Counter(iterator)
    total = sum(counter.values())
    header = f"{column} (rows={total:,}, unique={len(counter):,})"
    print(header)
    print("-" * len(header))

    most_common = [
        (value, count) for value, count in counter.most_common() if count >= min_count
    ][:top_k]
    if not most_common:
        print("No values meet the requested thresholds.\n")
        return

    print(f"Top {len(most_common)} values (min_count={min_count}):")
    max_len = max(len(value) for value, _ in most_common)
    for value, count in most_common:
        print(f"  {value:<{max_len}}  {count:>12,}")

    if search_terms:
        print()
        for term in search_terms:
            term_lower = term.lower()
            matches = [
                (value, count)
                for value, count in counter.items()
                if term_lower in value.lower() and count >= min_count
            ]
            if not matches:
                print(f"No values containing '{term}' with min_count={min_count}.")
                continue
            print(f"Values containing '{term}':")
            max_len = max(len(value) for value, _ in matches)
            for value, count in sorted(matches, key=lambda x: -x[1]):
                print(f"  {value:<{max_len}}  {count:>12,}")
        print()


def summarize_edge_types(
    path: str,
    top_k: int,
    min_count: int,
    node_categories: dict[str, str],
) -> None:
    """Aggregate (subject_category, predicate, object_category) triples."""
    counter = Counter()
    parquet_file = pq.ParquetFile(path)
    for rg_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            rg_index, columns=["subject", "predicate", "object"]
        )
        subjects = table["subject"].to_pylist()
        predicates = table["predicate"].to_pylist()
        objects = table["object"].to_pylist()
        for subj, pred, obj in zip(subjects, predicates, objects):
            subj_cat = node_categories.get(subj, "UNKNOWN")
            obj_cat = node_categories.get(obj, "UNKNOWN")
            counter[(subj_cat, pred, obj_cat)] += 1

    total = sum(counter.values())
    header = f"edge types (rows={total:,}, unique={len(counter):,})"
    print(header)
    print("-" * len(header))
    most_common = [
        (triplet, count) for triplet, count in counter.most_common() if count >= min_count
    ][:top_k]
    if not most_common:
        print("No edge types meet the requested thresholds.\n")
        return

    print(f"Top {len(most_common)} (subject_category, predicate, object_category):")
    max_len = max(
        len(" | ".join(triplet))
        for triplet, _ in most_common
    )
    for (subj_cat, pred, obj_cat), count in most_common:
        label = f"{subj_cat} | {pred} | {obj_cat}"
        print(f"  {label:<{max_len}}  {count:>12,}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize column frequencies in a KG edge parquet file."
    )
    parser.add_argument(
        "parquet_path",
        help="Path to the kg_edges.parquet file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top items to print (default: 20).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Only display entries appearing at least this many times (default: 1).",
    )
    parser.add_argument(
        "--search",
        nargs="+",
        default=None,
        help="Optional case-insensitive substrings to highlight in column values.",
    )
    parser.add_argument(
        "-c",
        "--column",
        action="append",
        default=["predicate"],
        help=(
            "Column name to summarize (default: predicate). "
            "Repeat to analyze multiple columns (e.g., -c subject_category -c object_category)."
        ),
    )
    parser.add_argument(
        "--nodes-dir",
        default="nodes",
        help="Directory containing node parquet files mapped by id/category (default: nodes).",
    )
    parser.add_argument(
        "--edge-types",
        action="store_true",
        help="Summarize top (subject_category, predicate, object_category) combinations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_categories = None
    needs_categories = any(
        column in {"subject_category", "object_category"} for column in args.column
    )
    needs_edge_categories = args.edge_types
    if needs_categories or needs_edge_categories:
        node_path = Path(args.nodes_dir)
        if not node_path.exists():
            raise SystemExit(
                f"Node directory {node_path} not found; needed for subject/object categories."
            )
        node_categories = load_node_categories(node_path)
        print(f"Loaded {len(node_categories):,} nodes for category lookup.")

    for column in args.column:
        summarize_column(
            args.parquet_path,
            column,
            top_k=args.top,
            min_count=args.min_count,
            search_terms=args.search or [],
            node_categories=node_categories,
        )

    if args.edge_types:
        if node_categories is None:
            raise SystemExit("edge-types requires node category data; none was loaded.")
        summarize_edge_types(
            args.parquet_path,
            top_k=args.top,
            min_count=args.min_count,
            node_categories=node_categories,
        )


if __name__ == "__main__":
    main()
