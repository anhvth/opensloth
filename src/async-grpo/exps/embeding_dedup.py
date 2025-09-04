from __future__ import annotations

from typing import List, Literal
from collections import defaultdict
import numpy as np
import faiss


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        # path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def filter_duplicated(
    matrix: np.ndarray,
    eps: float = 0.08,
    *,
    metric: Literal['cosine', 'l2'] = 'cosine',
    assume_normalized: bool = True,
    k_neighbors: int = 16,
    batch_size: int = 8192,
    hnsw_m: int = 16,
    ef_search: int = 64,
) -> List[List[int]]:
    """
    Group vectors that are 'too close' (within eps) and return index groups.
    Uses HNSW k-NN (batched) + union-find to avoid O(N^2) memory/time.
    """
    if matrix.ndim != 2:
        raise ValueError('matrix must be 2D [N, D]')

    n, d = matrix.shape
    if n == 0:
        return []

    xb = np.asarray(matrix, dtype=np.float32, order='C')

    if metric == 'cosine' and not assume_normalized:
        faiss.normalize_L2(xb)

    index = faiss.IndexHNSWFlat(d, hnsw_m)
    index.hnsw.efSearch = ef_search
    index.add(xb)

    if metric == 'cosine':
        def close_enough(d2: float) -> bool:
            # for unit vectors: 0.5 * ||u - v||^2 == cosine_distance
            return 0.5 * d2 <= eps
    else:
        e2 = eps * eps

        def close_enough(d2: float) -> bool:
            return d2 <= e2

    dsu = DisjointSet(n)
    k = max(2, k_neighbors)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        D, I = index.search(xb[start:end], k)
        for row in range(end - start):
            i = start + row
            ids = I[row]
            d2s = D[row]
            for j, d2 in zip(ids, d2s):
                if j < 0 or j == i:
                    continue
                if close_enough(float(d2)):
                    dsu.union(i, j)

    comp_to_members: dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comp_to_members[dsu.find(i)].append(i)

    groups = []
    for members in comp_to_members.values():
        members.sort()
        groups.append(members)

    groups.sort(key=lambda g: g[0])
    return groups
