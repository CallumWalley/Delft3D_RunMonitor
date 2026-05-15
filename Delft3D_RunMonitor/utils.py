import numpy as np

from scipy.interpolate import splprep, splev, interp1d

from sklearn.cluster import KMeans
from shapely.geometry import Polygon


def calculate_clean_centerline(
    points,
    num_clusters=20,
    num_segments=100,
    stiffness=50.0,
    dense_samples=2000,
):
    """
    Compute a smooth centerline with equally spaced points.

    Parameters
    ----------
    points : (N,2) array
        River mesh points.

    num_clusters : int
        Number of KMeans clusters used to collapse the river width.

    num_segments : int
        Number of equal-length segments in the final centerline.

    stiffness : float
        Spline smoothing parameter.

    dense_samples : int
        Number of dense spline samples used for arc-length reconstruction.

    Returns
    -------
    centerline : (num_segments+1, 2) ndarray
        Equally spaced centerline points.
    """

    #
    # 1. Collapse river width using KMeans
    #
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init=10,
    )

    cluster_centers = kmeans.fit(points).cluster_centers_

    #
    # 2. Order centers from one river end to the other
    #
    start_idx = np.argmin(cluster_centers[:, 0])

    unvisited = list(range(num_clusters))
    unvisited.remove(start_idx)

    ordered_centers = [cluster_centers[start_idx]]
    current_center = cluster_centers[start_idx]

    while unvisited:

        distances = np.linalg.norm(
            cluster_centers[unvisited] - current_center,
            axis=1,
        )

        nearest_idx = unvisited[np.argmin(distances)]

        ordered_centers.append(cluster_centers[nearest_idx])

        unvisited.remove(nearest_idx)

        current_center = cluster_centers[nearest_idx]

    ordered_centers = np.asarray(ordered_centers)

    #
    # 3. Fit smooth spline
    #
    pts = ordered_centers.T

    tck, _ = splprep(pts, s=stiffness)

    #
    # 4. Sample spline densely
    #
    u_dense = np.linspace(0.0, 1.0, dense_samples)

    x_dense, y_dense = splev(u_dense, tck)

    x_dense = np.asarray(x_dense)
    y_dense = np.asarray(y_dense)

    #
    # 5. Compute cumulative arc length
    #
    dx = np.diff(x_dense)
    dy = np.diff(y_dense)

    ds = np.sqrt(dx**2 + dy**2)

    s = np.concatenate([[0.0], np.cumsum(ds)])

    total_length = s[-1]

    #
    # 6. Equally spaced arc-length positions
    #
    s_uniform = np.linspace(
        0.0,
        total_length,
        num_segments + 1,
    )

    #
    # 7. Interpolate coordinates onto uniform arc length
    #
    fx = interp1d(s, x_dense)
    fy = interp1d(s, y_dense)

    x_uniform = fx(s_uniform)
    y_uniform = fy(s_uniform)

    centerline = np.column_stack([x_uniform, y_uniform])

    return centerline


def triangle_area(coords):
    """
    Compute polygon area from coordinates.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    return 0.5 * abs(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    )


def compute_clipped_volume(
    points,
    triangles,
    height,
    clip_polygon_coords,
):
    """
    Compute volume inside a polygonal region.

    Parameters
    ----------
    points : (npoints, 2) array
        Mesh node coordinates.

    triangles : (ncells, 3) int array
        Triangle connectivity.

    height : (ncells,) array
        Cell-centered height values.

    clip_polygon_coords : (npoly, 2) array
        Coordinates defining clipping polygon.

    Returns
    -------
    volume : float
    """

    clip_poly = Polygon(clip_polygon_coords)

    volume = 0.0

    for icell, tri in enumerate(triangles):

        tri_coords = points[tri]

        tri_poly = Polygon(tri_coords)

        inter = tri_poly.intersection(clip_poly)

        if inter.is_empty:
            continue

        area = inter.area

        volume += height[icell] * area

    return volume