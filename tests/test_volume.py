import numpy as np
import matplotlib.pyplot as plt
from triangle import Triangle

from Delft3D_RunMonitor import compute_clipped_volume, VolumeIntegrator
 
def test_simple():

    # ---------------------------------------------------------
    # 1. triangulated domain a square
    # ---------------------------------------------------------
    ns1 = 5
    ns = ns1 - 1
    ds = 1.0 / (ns1 - 1)
    x0, y0 = 0.3, 0.3
    x1, y1 = 0.7, 0.9
    clip_polygon_coords = np.array(
        [(x0, y0),
         (x1, y0),
         (x1, y1),
         (x0, y1)]

    )
    xyb = [(i*ds, 0.0) for i in range(ns)] + \
          [(1.0, i*ds) for i in range(ns)] + \
          [(1.0 - i*ds, 1.0) for i in range(ns)] + \
          [(0.0, 1.0 - i*ds) for i in range(ns)]
    n = len(xyb)
    markers = [1 for _ in range(n)]
    segs = [(i, i + 1) for i in range(n)] + [(n - 1, 0)]

    print(f'xyb = {xyb}')
    print(f'segs = {segs}')

    tri = Triangle()
    tri.set_points(xyb, markers=markers)
    tri.set_segments(segs)
    tri.triangulate(area=0.5, mode='pzq10eQ')

    points = np.asarray([xy[0] for xy in tri.get_points()])
    triangles = np.asarray([t[0] for t in tri.get_triangles()])

    print(f'points = {points}')
    print(f'triangles = {triangles}')
    print(f'tri.get_triangles() = {tri.get_triangles()}')

    ncells = triangles.shape[0]
    heights = np.ones((ncells,), float)

    exact_volume = (x1 - x0) * (y1 - y0)

    # try the slow method
    volume = compute_clipped_volume(points, triangles, heights, clip_polygon_coords)
    print(f'volume = {volume}')

    assert abs(volume - exact_volume) < 1.e-10

    # this should be faster
    vi = VolumeIntegrator(points, triangles, clip_polygon_coords)
    volume = vi.get_volume(heights)
    assert abs(volume - exact_volume) < 1.e-10


def test_2():

    # ---------------------------------------------------------
    # 1. triangulated domain a square
    # ---------------------------------------------------------
    ns1 = 5
    ns = ns1 - 1
    ds = 1.0 / (ns1 - 1)
    x0, y0 = -0.5, -1
    x1, y1 = 1.5, -1
    x2, y2 = 0.5, 1
    clip_polygon_coords = np.array(
        [
         (x0, y0),
         (x1, y1),
         (x2, y2),
        ]

    )

    # [0,1] x [0,1] square
    xyb = [(i*ds, 0.0) for i in range(ns)] + \
          [(1.0, i*ds) for i in range(ns)] + \
          [(1.0 - i*ds, 1.0) for i in range(ns)] + \
          [(0.0, 1.0 - i*ds) for i in range(ns)]
    n = len(xyb)
    markers = [1 for _ in range(n)]
    segs = [(i, i + 1) for i in range(n)] + [(n - 1, 0)]

    print(f'xyb = {xyb}')
    print(f'segs = {segs}')

    tri = Triangle()
    tri.set_points(xyb, markers=markers)
    tri.set_segments(segs)
    tri.triangulate(area=0.5, mode='pzq10eQ')

    points = np.asarray([xy[0] for xy in tri.get_points()])
    triangles = np.asarray([t[0] for t in tri.get_triangles()])

    print(f'points = {points}')
    print(f'triangles = {triangles}')
    print(f'tri.get_triangles() = {tri.get_triangles()}')

    ncells = triangles.shape[0]
    heights = np.ones((ncells,), float)

    exact_volume = 0.5

    # try the slow method
    volume = compute_clipped_volume(points, triangles, heights, clip_polygon_coords)
    print(f'volume = {volume}')

    assert abs(volume - exact_volume) < 1.e-10

    # this should be faster
    vi = VolumeIntegrator(points, triangles, clip_polygon_coords)
    volume = vi.get_volume(heights)
    assert abs(volume - exact_volume) < 1.e-10




