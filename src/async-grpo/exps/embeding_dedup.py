from __future__ import annotations
from llm_utils import VectorCache
from typing import List, Literal, Optional
from collections import defaultdict
import math
import numpy as np
import faiss
import tqdm
import pandas as pd


def _build_hnsw(
    xb: np.ndarray,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> faiss.Index:
    d = xb.shape[1]
    index = faiss.IndexHNSWFlat(d, m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(xb)
    return index


def _build_ivf(
    xb: np.ndarray,
    nlist: int,
    nprobe: int,
) -> faiss.Index:
    d = xb.shape[1]
    quant = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_L2)
    index.train(xb)
    index.add(xb)
    index.nprobe = nprobe
    return index


def filter_duplicated(
    matrix: np.ndarray,
    eps: float = 0.08,  # 1 means to vector is identical, 0 means orthogonal, 0.08 means angle ~ 28 degree
    *,
    metric: Literal["cosine", "l2"] = "cosine",
    assume_normalized: bool = True,
    k_neighbors: int = 32,
    batch_size: int = 16384,
    engine: Literal["hnsw", "ivf"] = "hnsw",
    # HNSW
    hnsw_m: int = 24,
    ef_search: Optional[int] = None,
    ef_construction: int = 120,
    # IVF
    nlist: Optional[int] = None,
    nprobe: int = 16,
    # Threads
    n_threads: Optional[int] = None,
) -> List[List[int]]:
    """
    Complete-link style grouping: a point joins a group only if it is
    within 'eps' of *every* member of that group. Prevents A--B--C chains.

    Returns list of groups (each is a list of row indices).
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D [N, D]")

    n, d = matrix.shape
    if n == 0:
        return []

    if n_threads is not None:
        faiss.omp_set_num_threads(n_threads)

    xb = np.asarray(matrix, dtype=np.float32, order="C")

    if metric == "cosine" and not assume_normalized:
        faiss.normalize_L2(xb)

    # Work in squared L2; convert eps accordingly for cosine on unit sphere.
    if metric == "cosine":
        # cos_dist = 0.5 * ||u - v||^2  (u,v unit)
        d2_thresh = 2.0 * eps
    else:
        d2_thresh = eps * eps

    # Build ANN index
    if engine == "hnsw":
        eff = max(k_neighbors * 2, 64) if ef_search is None else ef_search
        index = _build_hnsw(
            xb=xb, m=hnsw_m, ef_construction=ef_construction, ef_search=eff
        )
    else:
        if nlist is None:
            est = int(4.0 * math.sqrt(float(n)))
            nlist = max(64, min(65536, est))
        nlist = min(nlist, max(1, n // 8)) if n >= 8 else 1
        index = _build_ivf(xb=xb, nlist=nlist, nprobe=nprobe)

    groups: List[List[int]] = []
    point_to_group = np.full(n, -1, dtype=np.int64)

    def can_join(group_idx: int, v_idx: int) -> bool:
        members = groups[group_idx]
        if not members:
            return True
        v = xb[v_idx]
        # Distances to all members; require max <= d2_thresh
        mtx = xb[np.asarray(members)]
        d2s = ((mtx - v) ** 2).sum(axis=1)
        return bool(np.all(d2s <= d2_thresh))

    k = max(2, k_neighbors)

    # Process in batches for speed and memory bounds
    for start in tqdm.tqdm(range(0, n, batch_size), desc="Processing batches"):
        end = min(n, start + batch_size)
        D, I = index.search(xb[start:end], k)

        for row in range(end - start):
            i = start + row
            # Gather candidate groups from already-assigned neighbors
            neigh_ids = I[row]
            cand = set()
            for j in neigh_ids:
                if j < 0 or j == i:
                    continue
                g = int(point_to_group[j])
                if g != -1:
                    cand.add(g)

            chosen: Optional[int] = None
            best_score = math.inf

            # Try to join the "tightest" feasible group first
            for g in cand:
                if can_join(g, i):
                    # score = max distance to members (lower is better)
                    mtx = xb[np.asarray(groups[g])]
                    if mtx.size == 0:
                        score = 0.0
                    else:
                        d2s = ((mtx - xb[i]) ** 2).sum(axis=1)
                        score = float(d2s.max())
                    if score < best_score:
                        best_score = score
                        chosen = g

            if chosen is None:
                # Make a new group
                point_to_group[i] = len(groups)
                groups.append([i])
            else:
                point_to_group[i] = chosen
                groups[chosen].append(i)

    # Sort members and groups for stable output
    for g in groups:
        g.sort()
    groups.sort(key=lambda g: g[0])

    return groups


def group_by_embedding(
    df_or_csv_path: str,
    vector_service: VectorCache,
    column: str = "source",
    output_path: str = None,
    eps: float = 0.08,
):
    df = (
        pd.read_csv(df_or_csv_path)
        if isinstance(df_or_csv_path, str)
        else df_or_csv_path
    )
    assert column in df.columns, f"{column} not in {df.columns}"
    embeds = vector_service(df[column].tolist())
    dedup_ids = filter_duplicated(embeds, eps=eps)
    list_gs = sorted(dedup_ids, key=len, reverse=True)
    
    # Initialize group_id column
    df["group_id"] = None
    
    # Map array indices to DataFrame indices
    df_indices = df.index.tolist()
    for i, g in enumerate(list_gs):
        # Convert array indices to DataFrame indices
        df_group_indices = [df_indices[idx] for idx in g]
        df.loc[df_group_indices, "group_id"] = int(i)
    grouped = df.groupby("group_id")
    dfs = [group for _, group in sorted(grouped, key=lambda x: len(x[1]), reverse=True)]
    df = pd.concat(dfs)
    if output_path:
        df.to_csv(output_path, index=False)
        return None
    return df
