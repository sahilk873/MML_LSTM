"""
Load kg_edges.parquet via polypharmacy.kg and print a sample of (src, dst) triples.
When the parquet has a relation/predicate column, it is included as (src, relation, dst).
Optionally write the sample to CSV. Run from repo root with PYTHONPATH=. if needed.
"""
import argparse
import sys
from pathlib import Path

# Allow importing polypharmacy when run from repo root (e.g. PYTHONPATH=. python scripts/show_kg_triples.py ...)
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from polypharmacy import kg as kg_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a KG edge parquet (e.g. kg_edges.parquet) and show a sample of (src, dst) or (src, relation, dst) triples."
    )
    parser.add_argument(
        "--kg",
        default="kg_edges.parquet",
        help="Path to the parquet file (default: kg_edges.parquet).",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=30,
        help="Number of triples to show (default: 30).",
    )
    parser.add_argument(
        "--edge-src-col",
        default=None,
        help="Source column name; if not set, inferred from parquet columns.",
    )
    parser.add_argument(
        "--edge-dst-col",
        default=None,
        help="Destination column name; if not set, inferred from parquet columns.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional path to write the sample (or full edges) to CSV.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample n random rows instead of the first n.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="When used with --output, write the full edge list to CSV (ignore -n for the file).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edges = kg_lib.load_edges(
        args.kg,
        src_col=args.edge_src_col,
        dst_col=args.edge_dst_col,
        include_relation=True,
    )
    total = len(edges)
    if args.random:
        n = min(args.num, total)
        sample = edges.sample(n=n, random_state=42)
    else:
        sample = edges.head(args.num)
    print(f"Total edges: {total:,}")
    cols = list(sample.columns)
    if "relation" in cols:
        print(f"Showing {len(sample):,} triples (src, relation, dst):\n")
    else:
        print(f"Showing {len(sample):,} triples (src, dst):\n")
    print(sample.to_string(index=False))
    if args.output:
        out_df = edges if args.full else sample
        out_df.to_csv(args.output, index=False)
        print(f"\nWrote {len(out_df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
