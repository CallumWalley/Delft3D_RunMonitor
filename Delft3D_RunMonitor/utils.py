import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def compute_centerline(points, triangles, ds, k=6):
    """
    Robust centerline extraction from triangular mesh using graph geodesics.
    """

    # ---------------------------------------------------------
    # 1. triangle centroids
    # ---------------------------------------------------------
    tri_pts = points[triangles]
    centroids = tri_pts.mean(axis=1)

    n = len(centroids)

    # ---------------------------------------------------------
    # 2. build k-nearest-neighbour graph
    # ---------------------------------------------------------
    tree = cKDTree(centroids)
    dists, idxs = tree.query(centroids, k=12) #k=k+1)

    rows = []
    cols = []
    data = []

    for i in range(n):
        for j, d in zip(idxs[i][1:], dists[i][1:]):
            # keep only short edges (important!)
            if d < np.percentile(dists[:, 1:], 30):
                rows.append(i)
                cols.append(j)
                data.append(d)

    G = csr_matrix((data, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)

    # ---------------------------------------------------------
    # 3. approximate diameter endpoints
    # ---------------------------------------------------------
    d0 = shortest_path(G, directed=False, indices=0)
    a = np.argmax(d0)

    d1 = shortest_path(G, directed=False, indices=a)
    b = np.argmax(d1)

    # ---------------------------------------------------------
    # 4. shortest path between endpoints
    # ---------------------------------------------------------
    dist, pred = shortest_path(
        G,
        directed=False,
        indices=a,
        return_predecessors=True
    )

    # reconstruct path
    path = []
    cur = b
    while cur != a:
        path.append(cur)
        cur = pred[cur]
        if cur == -9999:
            raise RuntimeError("Path reconstruction failed")

    path.append(a)
    path = path[::-1]

    line = centroids[path]

    # ---------------------------------------------------------
    # 5. resample by arc-length
    # ---------------------------------------------------------
    seg = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
    s = np.concatenate([[0], np.cumsum(seg)])

    s_new = np.arange(0, s[-1], ds)

    x = np.interp(s_new, s, line[:, 0])
    y = np.interp(s_new, s, line[:, 1])

    return np.column_stack([x, y])
