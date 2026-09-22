import os
import tempfile
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# SPO triples, column names for our parquet (kushal)
SUBJECT_COL = "subject"
OBJECT_COL = "object"
PREDICATE_COL = "predicate"


# returns df with src/dst, fine for node2vec
def load_edges(
    path: str,
    src_col: Optional[str] = None,
    dst_col: Optional[str] = None,
    include_relation: bool = False,
) -> pd.DataFrame:
    """Load KG edges from parquet. Expects columns subject, object, and optionally predicate."""
    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Reading parquet requires pyarrow or fastparquet. "
            "Install one of them before running."
        ) from exc

    src_col = src_col if src_col is not None else SUBJECT_COL
    dst_col = dst_col if dst_col is not None else OBJECT_COL

    if src_col not in df.columns or dst_col not in df.columns:
        missing = [c for c in (src_col, dst_col) if c not in df.columns]
        raise ValueError(
            f"Edge columns not found: {missing}. Parquet columns: {list(df.columns)}."
        )

    rename = {src_col: "src", dst_col: "dst"}
    if include_relation:
        rel_col = PREDICATE_COL if PREDICATE_COL in df.columns else None
        if rel_col is not None:
            out_cols = [src_col, dst_col, rel_col]
            rename[rel_col] = "relation"
        else:
            out_cols = [src_col, dst_col]
    else:
        out_cols = [src_col, dst_col]
    return df[out_cols].rename(columns=rename)


def sample_edges(
    path: str,
    n: int,
    random: bool = False,
    src_col: Optional[str] = None,
    dst_col: Optional[str] = None,
    include_relation: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Load edges from parquet and return the first n rows or a random sample of n."""
    edges = load_edges(
        path,
        src_col=src_col,
        dst_col=dst_col,
        include_relation=include_relation,
    )
    if random:
        return edges.sample(n=min(n, len(edges)), random_state=seed)
    return edges.head(n)


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
    backend: str = "auto",
) -> Tuple[List[str], np.ndarray]:
    edges = edges.dropna(subset=["src", "dst"]).copy()
    edges["src"] = edges["src"].astype(str).str.strip()
    edges["dst"] = edges["dst"].astype(str).str.strip()
    edges = edges[(edges["src"] != "") & (edges["dst"] != "")]
    edge_count = len(edges)
    unique_nodes = pd.unique(
        pd.concat([edges["src"], edges["dst"]], ignore_index=True)
    ).size
    print(
        "KG node2vec setup | edges="
        f"{edge_count} nodes={unique_nodes} dim={embedding_dim} "
        f"walk_length={walk_length} num_walks={num_walks} "
        f"p={p} q={q} window={context_window} workers={workers} backend={backend}"
    )

    def run_with_pecanpy() -> Tuple[List[str], np.ndarray]:
        from gensim.models import Word2Vec

        # see notes doc for pecanpy impl
        try:
            from pecanpy import SparseOTF
            sparse_class = SparseOTF
        except ImportError:
            import pecanpy
            sparse_class = getattr(pecanpy, "SparseOTF", None)
            if sparse_class is None:
                for name in dir(pecanpy):
                    if name.endswith("OTF"):
                        sparse_class = getattr(pecanpy, name)
                        break
            if sparse_class is None:
                raise ImportError("PecanPy OTF class not found.")

        edge_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".edgelist", delete=False
            ) as handle:
                skipped = 0
                for src, dst in edges[["src", "dst"]].itertuples(index=False, name=None):
                    if not src or not dst:
                        skipped += 1
                        continue
                    handle.write(f"{src} {dst}\n")
                edge_path = handle.name
            if skipped:
                print(f"KG edgelist export: skipped {skipped} invalid edges")

            print(f"KG node2vec backend: pecanpy {sparse_class.__name__}")
            start = time.time()
            graph = sparse_class(p=p, q=q, workers=workers, verbose=False)
            graph.read_edg(edge_path, weighted=False, directed=False, delimiter=" ")
            if hasattr(graph, "preprocess_transition_probs"):
                graph.preprocess_transition_probs()

            walks = graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)
            model = Word2Vec(
                sentences=walks,
                vector_size=embedding_dim,
                window=context_window,
                min_count=min_count,
                sg=1,
                workers=workers,
                seed=seed,
            )
            node_ids = list(model.wv.index_to_key)
            vectors = model.wv.vectors
            print(
                "KG node2vec complete (pecanpy) | "
                f"nodes_embedded={len(node_ids)} in {time.time() - start:.1f}s"
            )
            return node_ids, vectors
        finally:
            if edge_path:
                os.remove(edge_path)

    def run_with_node2vec() -> Tuple[List[str], np.ndarray]:
        import networkx as nx
        from node2vec import Node2Vec

        print("KG node2vec backend: node2vec (networkx)")
        start = time.time()
        graph = nx.from_pandas_edgelist(edges, source="src", target="dst")
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
        print(
            "KG node2vec complete (node2vec) | "
            f"nodes_embedded={len(node_ids)} in {time.time() - start:.1f}s"
        )
        return node_ids, vectors

    if backend == "pecanpy":
        return run_with_pecanpy()
    if backend == "node2vec":
        return run_with_node2vec()

    try:
        return run_with_pecanpy()
    except ImportError as exc:
        print(f"KG node2vec backend: pecanpy unavailable ({exc}); falling back to node2vec.")
        return run_with_node2vec()


def extract_kg_nodes(edges: pd.DataFrame) -> List[str]:
    return pd.unique(pd.concat([edges["src"], edges["dst"]], ignore_index=True)).tolist()


def keep_edges_incident_to_nodes(edges: pd.DataFrame, nodes: Iterable[str]) -> pd.DataFrame:
    """Keep only edges that have at least one endpoint in `nodes`."""
    node_set = set(nodes)
    mask = edges["src"].isin(node_set) | edges["dst"].isin(node_set)
    return edges[mask].copy()


# alias for backward compat
prune_edges_to_nodes = keep_edges_incident_to_nodes


def expand_node_set(
    edges: pd.DataFrame,
    initial_nodes: Iterable[str],
    hops: int,
    max_nodes: Optional[int] = None,
) -> Tuple[Set[str], List[Tuple[int, int, int]], bool]:
    """Expand by walking up to `hops` from initial_nodes. Returns (expanded_ids, hop_logs, truncated)."""
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
    backend: str = "auto",
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
        backend=backend,
    )
    if cache_path:
        np.savez_compressed(cache_path, node_ids=np.array(node_ids), embeddings=vectors)
    return node_ids, vectors


def load_precomputed_embeddings(
    embeddings_path: str, node_ids_path: Optional[str] = None
) -> Tuple[List[str], np.ndarray]:
    if embeddings_path.endswith(".npz"):
        data = np.load(embeddings_path, allow_pickle=True)
        node_ids = data["node_ids"].tolist()
        vectors = data["embeddings"]
    elif embeddings_path.endswith(".npy"):
        if node_ids_path is None:
            raise ValueError("node_ids_path is required when using .npy embeddings.")
        vectors = np.load(embeddings_path)
        node_ids = np.load(node_ids_path, allow_pickle=True).tolist()
    else:
        raise ValueError(f"Unsupported embedding file: {embeddings_path}")
    if len(node_ids) != vectors.shape[0]:
        raise ValueError(
            "Embedding count mismatch: "
            f"{len(node_ids)} node IDs vs {vectors.shape[0]} vectors."
        )
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
