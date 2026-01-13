import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


EDGE_COLUMN_CANDIDATES: List[Tuple[str, str]] = [
    ("subject", "object"),
    ("source", "target"),
    ("src", "dst"),
    ("head", "tail"),
    ("node1", "node2"),
]


def _infer_edge_columns(columns: Iterable[str]) -> Tuple[str, str]:
    lower_map = {col.lower(): col for col in columns}
    for left, right in EDGE_COLUMN_CANDIDATES:
        if left in lower_map and right in lower_map:
            return lower_map[left], lower_map[right]
    raise ValueError(
        "Unable to infer edge columns. Provide --edge-src-col and --edge-dst-col."
    )


def load_edges(path: str, src_col: Optional[str], dst_col: Optional[str]) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Reading parquet requires pyarrow or fastparquet. "
            "Install one of them before running."
        ) from exc

    if src_col is None or dst_col is None:
        inferred_src, inferred_dst = _infer_edge_columns(df.columns)
        src_col = src_col or inferred_src
        dst_col = dst_col or inferred_dst

    if src_col not in df.columns or dst_col not in df.columns:
        raise ValueError(f"Edge columns not found: {src_col}, {dst_col}")

    return df[[src_col, dst_col]].rename(columns={src_col: "src", dst_col: "dst"})


def build_node2vec_embeddings(
    edges: pd.DataFrame,
    embedding_dim: int,
    walk_length: int,
    num_walks: int,
    p: float,
    q: float,
    context_window: int,
    min_count: int,
    workers: int,
    seed: int,
) -> Tuple[List[str], np.ndarray]:
    import networkx as nx
    from node2vec import Node2Vec

    graph = nx.from_pandas_edgelist(edges, source="src", target="dst")
    # Node2Vec embeds all KG nodes into a shared vector space.
    node2vec = Node2Vec(
        graph,
        dimensions=embedding_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
    )
    model = node2vec.fit(
        window=context_window, min_count=min_count, batch_words=128, seed=seed
    )
    node_ids = list(model.wv.index_to_key)
    vectors = model.wv.vectors
    return node_ids, vectors


def extract_kg_nodes(edges: pd.DataFrame) -> List[str]:
    return pd.unique(pd.concat([edges["src"], edges["dst"]], ignore_index=True)).tolist()


def prune_edges_to_nodes(edges: pd.DataFrame, nodes: Iterable[str]) -> pd.DataFrame:
    node_set = set(nodes)
    mask = edges["src"].isin(node_set) | edges["dst"].isin(node_set)
    return edges[mask].copy()


def expand_node_set(
    edges: pd.DataFrame,
    initial_nodes: Iterable[str],
    hops: int,
    max_nodes: Optional[int] = None,
) -> Tuple[Set[str], List[Tuple[int, int, int]], bool]:
    """Expand the node set by walking up to `hops` in the KG adjacency.

    Args:
        edges: DataFrame with `src` and `dst` columns.
        initial_nodes: Starting node IDs from the dataset.
        hops: Number of hops to expand.
        max_nodes: Optional cap; expansion stops if exceeded.

    Returns:
        expanded_nodes: Set of node IDs after expansion.
        hop_logs: List of tuples (hop_number, new_nodes_added, cumulative_nodes).
        truncated: True if expansion stopped early due to max_nodes.

    Example:
        >>> df = pd.DataFrame({'src': ['A', 'B'], 'dst': ['B', 'C']})
        >>> expand_node_set(df, {'A'}, hops=1)[0]
        {'A', 'B'}
        >>> expand_node_set(df, {'A'}, hops=2)[0]
        {'A', 'B', 'C'}
    """
    if hops <= 0:
        return set(initial_nodes), [], False

    src_arr = edges["src"].to_numpy()
    dst_arr = edges["dst"].to_numpy()
    expanded = set(initial_nodes)
    frontier = set(initial_nodes)
    hop_logs: List[Tuple[int, int, int]] = []
    truncated = False

    for hop in range(1, hops + 1):
        if not frontier:
            break
        frontier_arr = np.array(list(frontier), dtype=object)
        src_mask = np.isin(src_arr, frontier_arr)
        dst_mask = np.isin(dst_arr, frontier_arr)

        neighbor_parts = []
        if src_mask.any():
            neighbor_parts.append(dst_arr[src_mask])
        if dst_mask.any():
            neighbor_parts.append(src_arr[dst_mask])
        if not neighbor_parts:
            break
        neighbors = np.concatenate(neighbor_parts)

        unique_neighbors = np.unique(neighbors)
        new_nodes = set(unique_neighbors) - expanded
        if not new_nodes:
            break

        expanded.update(new_nodes)
        hop_logs.append((hop, len(new_nodes), len(expanded)))

        if max_nodes is not None and len(expanded) > max_nodes:
            truncated = True
            break

        frontier = new_nodes

    return expanded, hop_logs, truncated


def load_or_build_kg_embeddings(
    kg_path: str,
    cache_path: str,
    embedding_dim: int,
    walk_length: int,
    num_walks: int,
    p: float,
    q: float,
    context_window: int,
    min_count: int,
    workers: int,
    seed: int,
    src_col: Optional[str],
    dst_col: Optional[str],
    edges: Optional[pd.DataFrame] = None,
) -> Tuple[List[str], np.ndarray]:
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        node_ids = data["node_ids"].tolist()
        vectors = data["embeddings"]
        return node_ids, vectors

    if edges is None:
        edges = load_edges(kg_path, src_col=src_col, dst_col=dst_col)
    node_ids, vectors = build_node2vec_embeddings(
        edges=edges,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        context_window=context_window,
        min_count=min_count,
        workers=workers,
        seed=seed,
    )
    if cache_path:
        np.savez_compressed(cache_path, node_ids=np.array(node_ids), embeddings=vectors)
    return node_ids, vectors


def build_entity_embedding(
    entity_ids: List[str],
    node_to_idx: Dict[str, int],
    node_embeddings: np.ndarray,
    embedding_dim: int,
    rng: np.random.RandomState,
    pad_idx: Optional[int] = None,
) -> np.ndarray:
    embeddings = np.zeros((len(entity_ids), embedding_dim), dtype=np.float32)
    for idx, entity_id in enumerate(entity_ids):
        if pad_idx is not None and idx == pad_idx:
            continue
        node_idx = node_to_idx.get(entity_id)
        if node_idx is not None:
            embeddings[idx] = node_embeddings[node_idx]
        else:
            embeddings[idx] = rng.normal(scale=0.1, size=embedding_dim)
    return embeddings
