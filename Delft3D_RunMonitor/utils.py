import numpy as np


def compute_centerline(points, triangles, ds):
    """
    Compute an approximate river centerline from a triangular mesh.

    Args:
        points:
            Array of node coordinates of shape (npoints, 2).

        triangles:
            Triangle connectivity array of shape (ntri, 3).

        ds:
            Desired spacing between centerline points.

    Returns:
        List of (x, y) tuples representing the centerline.
    """

    # ---------------------------------------------------------
    # Compute triangle centroids
    # ---------------------------------------------------------

    tri_pts = points[triangles]          # (ntri, 3, 2)
    centroids = tri_pts.mean(axis=1)     # (ntri, 2)

    # ---------------------------------------------------------
    # Estimate dominant river direction using PCA
    # ---------------------------------------------------------

    mean_xy = centroids.mean(axis=0)

    X = centroids - mean_xy

    # Singular value decomposition
    _, _, vh = np.linalg.svd(X, full_matrices=False)

    # Main river direction
    tangent = vh[0]

    # Transverse direction
    normal = np.array([-tangent[1], tangent[0]])

    # ---------------------------------------------------------
    # Project centroids into curvilinear coordinates
    # ---------------------------------------------------------

    s = X @ tangent
    n = X @ normal

    # ---------------------------------------------------------
    # Build bins along the river axis
    # ---------------------------------------------------------

    smin = s.min()
    smax = s.max()

    sbins = np.arange(smin, smax + ds, ds)

    centerline = []

    # ---------------------------------------------------------
    # Average transverse location in each bin
    # ---------------------------------------------------------

    for s0 in sbins:

        mask = (s >= s0) & (s < s0 + ds)

        if np.count_nonzero(mask) < 3:
            continue

        sm = np.mean(s[mask])
        nm = np.mean(n[mask])

        # Convert back to x,y
        xy = mean_xy + sm * tangent + nm * normal

        centerline.append((xy[0], xy[1]))

    return centerline