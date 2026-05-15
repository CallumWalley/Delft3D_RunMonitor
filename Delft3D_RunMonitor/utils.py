import numpy as np

from scipy.interpolate import splprep, splev, interp1d

from sklearn.cluster import KMeans
from shapely.geometry import LineString, Polygon

def perpendicular_cross_section(
    point,
    centerline,
    points,
    triangles,
):
    """
    Given a point on a centerline and a triangulation,
    compute the two intersection points of the perpendicular line
    with the mesh boundary.

    Returns
    -------
    n : (2,) ndarray
        Unit normal vector to centerline at the point.

    t : (2,) ndarray
        Scalar parameters along the normal direction:
        p0 + t[i] * n

    x : (2,2) ndarray
        The two intersection points in physical space.
    """

    #
    # 1. Find closest centerline segment
    #
    centerline = np.asarray(centerline)

    d = np.linalg.norm(centerline - point, axis=1)
    i = np.argmin(d)

    if i == 0:
        i0, i1 = 0, 1
    elif i == len(centerline) - 1:
        i0, i1 = i - 1, i
    else:
        if d[i + 1] < d[i - 1]:
            i0, i1 = i, i + 1
        else:
            i0, i1 = i - 1, i

    p0 = centerline[i0]
    p1 = centerline[i1]

    #
    # 2. Tangent and normal
    #
    tangent = p1 - p0
    tangent = tangent / np.linalg.norm(tangent)

    normal = np.array([-tangent[1], tangent[0]])

    #
    # 3. Build boundary polygon from triangulation
    #
    # Extract unique boundary edges
    edges = {}

    for tri in triangles:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = tuple(sorted((a, b)))
            edges[key] = edges.get(key, 0) + 1

    boundary_edges = [e for e, c in edges.items() if c == 1]

    #
    # 4. Convert boundary to shapely segments
    #
    boundary_lines = [
        LineString([points[i], points[j]])
        for i, j in boundary_edges
    ]

    #
    # 5. Shoot perpendicular line
    #
    L = 1e6  # large enough to cross domain

    line = LineString([
        point - L * normal,
        point + L * normal
    ])

    #
    # 6. Intersect with boundary
    #
    intersections = []

    for seg in boundary_lines:
        inter = line.intersection(seg)

        if inter.is_empty:
            continue

        if inter.geom_type == "Point":
            intersections.append(inter)

        elif inter.geom_type == "MultiPoint":
            intersections.extend(list(inter.geoms))

    #
    # 7. Extract two points
    #
    if len(intersections) < 2:
        raise RuntimeError("Could not find two boundary intersections")

    pts = np.array([[p.x, p.y] for p in intersections])

    #
    # 8. Project onto normal axis to get scalar parameters
    #
    rel = pts - point
    t = rel @ normal

    #
    # 9. Sort consistently
    #
    order = np.argsort(t)

    pts = pts[order]
    t = t[order]

    return normal, t, pts


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