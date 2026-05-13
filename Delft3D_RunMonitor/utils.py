import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import splprep, splev
from sklearn.cluster import KMeans


import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import splprep, splev

def calculate_clean_centerline(points, num_clusters=20, num_output_points=100, stiffness=50.0):
    """
    Collapses a wide river mesh into a central thread using K-Means, 
    then fits a smooth, loop-free spline.
    """
    # 1. Use K-Means to find points directly in the middle of the river channel
    # This collapses the width of the river into a single centerline thread
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_centers = kmeans.fit(points).cluster_centers_
    
    # 2. Sort the cluster centers from one end of the river to the other
    # We find the leftmost cluster center to start the sequence
    start_idx = np.argmin(cluster_centers[:, 0])
    
    unvisited = list(range(num_clusters))
    unvisited.remove(start_idx)
    ordered_centers = [cluster_centers[start_idx]]
    current_center = cluster_centers[start_idx]
    
    while unvisited:
        # Find the nearest next cluster center
        distances = np.linalg.norm(cluster_centers[unvisited] - current_center, axis=1)
        nearest_idx = unvisited[np.argmin(distances)]
        
        ordered_centers.append(cluster_centers[nearest_idx])
        unvisited.remove(nearest_idx)
        current_center = cluster_centers[nearest_idx]
        
    ordered_centers = np.array(ordered_centers)
    
    # 3. Fit a stiff B-spline through the clean center thread
    # Because the points are already in the middle, loops are impossible
    pts = ordered_centers.T
    tck, u = splprep(pts, s=stiffness)
    
    # 4. Generate the final continuous centerline
    u_new = np.linspace(0, 1, num_output_points)
    centerline = splev(u_new, tck)
    
    return np.array(centerline).T


def sort_points_by_proximity(points):
    """
    Orders unordered cloud points sequentially from one end to the other
    using a nearest-neighbor walking path.
    """
    num_pts = len(points)
    unvisited = set(range(num_pts))
    
    # 1. Fit a Nearest Neighbors model to find closest points
    nn = NearestNeighbors(n_neighbors=min(5, num_pts)).fit(points)
    
    # 2. Find an endpoint to start the walk (point with the minimum X coordinate)
    start_idx = np.argmin(points[:, 0]) 
    
    ordered_indices = [start_idx]
    unvisited.remove(start_idx)
    current_idx = start_idx
    
    # 3. Walk through the cloud point by point
    while unvisited:
        # Find closest neighbors
        distances, indices = nn.kneighbors([points[current_idx]], n_neighbors=num_pts)
        
        # Select the closest neighbor that hasn't been visited yet
        next_idx = None
        for idx in indices[0]:
            if idx in unvisited:
                next_idx = idx
                break
        
        # If no local neighbors are unvisited, jump to the absolute closest unvisited point
        if next_idx is None:
            next_idx = min(unvisited, key=lambda i: np.linalg.norm(points[i] - points[current_idx]))
            
        ordered_indices.append(next_idx)
        unvisited.remove(next_idx)
        current_idx = next_idx
        
    return points[ordered_indices]

def calculate_stiff_river_centerline(points, num_points=100, stiffness=500.0):
    """
    Sorts points topologically along the river path, then fits a stiff spline.
    """
    # 1. Cleanly order the points from end to end
    ordered_points = sort_points_by_proximity(points)
    
    # 2. Transpose for splprep requirement: shape (2, N)
    pts = ordered_points.T
    
    # 3. Fit the parametric B-Spline.
    # Adjust 's' (stiffness) to smoothly average out the width of the banks.
    tck, u = splprep(pts, s=stiffness)
    
    # 4. Generate the continuous, stiff path
    u_new = np.linspace(0, 1, num_points)
    centerline = splev(u_new, tck)
    
    return np.array(centerline).T



def calculate_universal_centerline(points, num_points=100, stiffness=1000.0):
    """
    Computes a continuous, stiff centerline for both straight and curved rivers
    using PCA projection alignment and independent structural splines.
    
    Args:
        points (ndarray): Shape (N, 2) cloud points.
        num_points (int): Number of points in output curve.
        stiffness (float): Smoothing factor. Higher values force the line 
                           to be straighter/stiffer.
    """
    # 1. Use PCA to find the primary flow direction of the river
    pca = PCA(n_components=2)
    projected_points = pca.fit_transform(points)
    
    # Primary axis (flow direction) is column 0, cross-river axis is column 1
    flow_axis = projected_points[:, 0]
    cross_axis = projected_points[:, 1]
    
    # 2. Sort points chronologically along the flow direction
    sort_idx = np.argsort(flow_axis)
    sorted_flow = flow_axis[sort_idx]
    sorted_cross = cross_axis[sort_idx]
    
    # 3. Deduplicate matching flow coordinates to prevent spline calculation errors
    _, unique_idx = np.unique(sorted_flow, return_index=True)
    clean_flow = sorted_flow[unique_idx]
    clean_cross = sorted_cross[unique_idx]
    
    # 4. Fit an infinitely continuous spline over the sorted, projected data
    # 's' controls stiffness. If s is massive, this becomes a perfect linear regression.
    spline = UnivariateSpline(clean_flow, clean_cross, s=stiffness)
    
    # 5. Generate smooth path along the main axis
    flow_eval = np.linspace(clean_flow.min(), clean_flow.max(), num_points)
    cross_eval = spline(flow_eval)
    
    # 6. Reconstruct points back into original (X, Y) space
    eval_projected = np.vstack((flow_eval, cross_eval)).T
    centerline = pca.inverse_transform(eval_projected)
    
    return centerline


def calculate_river_centerline(points, num_points=100, stiffness=10.0):
    """
    Fits a stiff B-spline (river center) through a set of points.
    
    Args:
        points (ndarray): Shape (N, 2) array of (x, y) coordinates.
        num_points (int): Number of points on the resulting centerline.
        stiffness (float): Smoothing parameter 's'. Higher means stiffer.
                           If 0, it passes through all points.
                           
    Returns:
        ndarray: Smoothed (x, y) center points.
    """
    # Splprep expects shape (2, N)
    pts = points.T
    
    # splprep fits a B-spline. The s parameter controls stiffness.
    # It minimizes: sum((dist(points, curve))^2) + s * integral(curvature^2)
    tck, u = splprep(pts, s=stiffness)
    
    # Generate points along the fitted curve
    u_new = np.linspace(0, 1, num_points)
    centerline = splev(u_new, tck)
    
    # Return as (num_points, 2)
    return np.array(centerline).T



def principal_curve(
    points,
    n_iter=20,
    smoothing=0.01,
    n_curve_points=2000,
    ):
    """
    Robust principal curve fit for 2D point clouds.

    Parameters
    ----------
    points : (N,2) ndarray
        Input point cloud.

    n_iter : int
        Number of iterations.

    smoothing : float
        Relative smoothing parameter.
        Larger -> smoother curve.

    n_curve_points : int
        Number of dense samples used for projection.

    Returns
    -------
    curve : callable
        curve(t) -> (x,y)

    t : ndarray
        Parameter values for each input point.

    projected : ndarray
        Projection of points onto curve.
    """

    points = np.asarray(points)

    #
    # --- PCA initialization ---
    #

    pca = PCA(n_components=1)

    t = pca.fit_transform(points).ravel()

    t = (t - t.min()) / (t.max() - t.min())

    projected = points.copy()

    for _ in range(n_iter):

        #
        # --- sort by parameter ---
        #

        order = np.argsort(t)

        ts = t[order]
        xs = points[order, 0]
        ys = points[order, 1]

        #
        # --- remove duplicate t values ---
        #

        eps = 1e-10

        keep = np.concatenate([
            [True],
            np.diff(ts) > eps
        ])

        ts = ts[keep]
        xs = xs[keep]
        ys = ys[keep]

        #
        # --- spline fit ---
        #

        # scale smoothing with number of points
        s = smoothing * len(ts)

        tck, _ = splprep(
            [xs, ys],
            u=ts,
            s=s,
            k=3,
        )

        #
        # --- dense curve sampling ---
        #

        t_dense = np.linspace(0, 1, n_curve_points)

        x_dense, y_dense = splev(t_dense, tck)

        curve_dense = np.column_stack([x_dense, y_dense])

        #
        # --- nearest projection ---
        #

        nbrs = NearestNeighbors(n_neighbors=1)

        nbrs.fit(curve_dense)

        _, idx = nbrs.kneighbors(points)

        idx = idx.ravel()

        projected = curve_dense[idx]

        t_new = t_dense[idx]

        #
        # --- convergence ---
        #

        delta = np.mean(np.abs(t_new - t))

        t = t_new

        if delta < 1e-5:
            break

    #
    # --- return callable ---
    #

    def curve(tt):

        tt = np.asarray(tt)

        x, y = splev(tt, tck)

        return np.column_stack([x, y])

    return curve, t, projected


def compute_centerline(points, triangles, ds, k=10):
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
    dists, idxs = tree.query(centroids, k=k+1)

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
