import numpy as np
import matplotlib.pyplot as plt
from triangle import Triangle
import time

from Delft3D_RunMonitor import FluxIntegrator
 
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

    # extract points, get_points returns a list of tuples (point, marker)
    points = np.asarray([xy[0] for xy in tri.get_points()])
    # extract triangles, get_triangles returns a list of tuples (triangle, marker)
    triangles = np.asarray([t[0] for t in tri.get_triangles()])
    # extract edges, get_edges returns a list of tuples (edge, marker)
    edges = np.asarray([e[0] for e in tri.get_edges()])

    print(f'points = {points}')
    print(f'triangles = {triangles}')
    print(f'edges = {edges}')

    p0 = (0.5, 0.0)
    p1 = (0.5, 1.0)
    fi = FluxIntegrator(points, triangles, edges, p0=p0, p1=p1)

    # set flux values on edge based on a potential field phi = y
    def phi(x, y):
        return y

    edge_values = np.zeros((edges.shape[0],), float)
    for iaib in edges:
        ia, ib = iaib
        x_a, y_a = points[ia]
        x_b, y_b = points[ib]
        edge_id = fi.nodes_edge[tuple(iaib)]
        # edge values are the difference in potential across the edge, phi_b - phi_a
        edge_values[edge_id] = phi(x_b, y_b) - phi(x_a, y_a)

    # compute the flux across the line segment
    flux = fi.get_flux(edge_values)
    print(f'flux = {flux}')

    exact_flux = phi(p1[0], p1[1]) - phi(p0[0], p0[1])
    assert abs(flux - exact_flux) < 1.e-10


def test_irregular():

    # boundary points
    xyb = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (1.0, 0.6), (0.0, 0.5),]
    n = len(xyb)
    # markers for boundary points, not used in this test but required by Triangle
    markers = [1 for _ in range(n)]
    # segments connecting the boundary points, must close the loop
    segs = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]

    # triangluate the domain, mode 'pzq10eQ' means:
    # p: triangulate a Planar Straight Line Graph
    # z: number all items starting from zero
    # q10: quality mesh with minimum angle of 10 degrees
    # e: output edge list
    # Q: quiet mode, no terminal output
    tri = Triangle()
    tri.set_points(xyb, markers=markers)
    tri.set_segments(segs)
    tri.triangulate(area=0.5, mode='pzq10eQ')

    points = np.asarray([xy[0] for xy in tri.get_points()])
    triangles = np.asarray([t[0] for t in tri.get_triangles()])
    edges = np.asarray([e[0] for e in tri.get_edges()]) 

    # create a flux integrator for the line segment
    p0 = (0.0, 0.0)
    p1 = (1.0, 0.6)
    fi = FluxIntegrator(points, triangles, edges, p0=p0, p1=p1)

    # print(f'points = {points}')
    # print(f'triangles = {triangles}')
    # plt.figure()
    # plt.triplot(points[:, 0], points[:, 1], triangles)
    # plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=2)
    # plt.show()

    # set flux values on edge based on a potential field phi = y
    def phi(x, y):
        return y
    
    edge_values = np.zeros((edges.shape[0],), float)
    for iaib in edges:
        ia, ib = iaib
        x_a, y_a = points[ia]
        x_b, y_b = points[ib]
        edge_id = fi.nodes_edge[tuple(iaib)]
        edge_values[edge_id] = phi(x_b, y_b) - phi(x_a, y_a)

    # compute the flux across the line segment
    flux = fi.get_flux(edge_values)
    print(f'flux = {flux}')

    exact_flux = phi(p1[0], p1[1]) - phi(p0[0], p0[1])
    assert abs(flux - exact_flux) < 1.e-10


def test_irregular2():

    # boundary points
    xyb = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (1.0, 0.6), (0.0, 0.5),]
    n = len(xyb)
    # markers for boundary points, not used in this test but required by Triangle
    markers = [1 for _ in range(n)]
    # segments connecting the boundary points, must close the loop
    segs = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]

    # triangluate the domain, mode 'pzq10eQ' means:
    # p: triangulate a Planar Straight Line Graph
    # z: number all items starting from zero
    # q10: quality mesh with minimum angle of 10 degrees
    # e: output edge list
    # Q: quiet mode, no terminal output
    tri = Triangle()
    tri.set_points(xyb, markers=markers)
    tri.set_segments(segs)
    tri.triangulate(area=0.5, mode='pzq10eQ')

    points = np.asarray([xy[0] for xy in tri.get_points()])
    triangles = np.asarray([t[0] for t in tri.get_triangles()])
    edges = np.asarray([e[0] for e in tri.get_edges()]) 

    # create a flux integrator for the line segment
    p0 = (0.0, 0.0)
    p1 = (2.0, 0.5)
    fi = FluxIntegrator(points, triangles, edges, p0=p0, p1=p1)

    # print(f'points = {points}')
    # print(f'triangles = {triangles}')
    # plt.figure()
    # plt.triplot(points[:, 0], points[:, 1], triangles)
    # plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=2)
    # plt.show()

    # set flux values on edge based on a potential field phi = y
    def phi(x, y):
        return y
    
    edge_values = np.zeros((edges.shape[0],), float)
    for iaib in edges:
        ia, ib = iaib
        x_a, y_a = points[ia]
        x_b, y_b = points[ib]
        edge_id = fi.nodes_edge[tuple(iaib)]
        edge_values[edge_id] = phi(x_b, y_b) - phi(x_a, y_a)

    # compute the flux across the line segment
    flux = fi.get_flux(edge_values)
    print(f'flux = {flux}')

    exact_flux = phi(p1[0], p1[1]) - phi(p0[0], p0[1])
    print(f'exact_flux = {exact_flux}')
    assert abs(flux - exact_flux) < 1.e-10

def test_river():

    # boundary points
    xyb = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.6), (3.0, 1.0), (5.0, 1.0), (5.0, 2.0), (4.0, 2.0), (2.0, 3.0), (0.0, 0.5),]
    n = len(xyb)
    # markers for boundary points, not used in this test but required by Triangle
    markers = [1 for _ in range(n)]
    # segments connecting the boundary points, must close the loop
    segs = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]

    # triangluate the domain, mode 'pzq10eQ' means:
    # p: triangulate a Planar Straight Line Graph
    # z: number all items starting from zero
    # q10: quality mesh with minimum angle of 10 degrees
    # e: output edge list
    # Q: quiet mode, no terminal output
    tri = Triangle()
    tri.set_points(xyb, markers=markers)
    tri.set_segments(segs)
    tri.triangulate(area=0.5, mode='pzq10eQ')

    points = np.asarray([xy[0] for xy in tri.get_points()])
    triangles = np.asarray([t[0] for t in tri.get_triangles()])
    edges = np.asarray([e[0] for e in tri.get_edges()]) 

    # create a flux integrator for the line segment
    p0 = (0.0, 0.0)
    p1 = (2.0, 3.0)
    fi = FluxIntegrator(points, triangles, edges, p0=p0, p1=p1)

    # print(f'points = {points}')
    # print(f'triangles = {triangles}')
    # plt.figure()
    # plt.triplot(points[:, 0], points[:, 1], triangles)
    # plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=2)
    # plt.show()

    # set flux values on edge based on a potential field phi = y
    def phi(x, y):
        return y
    
    edge_values = np.zeros((edges.shape[0],), float)
    for iaib in edges:
        ia, ib = iaib
        x_a, y_a = points[ia]
        x_b, y_b = points[ib]
        edge_id = fi.nodes_edge[tuple(iaib)]
        edge_values[edge_id] = phi(x_b, y_b) - phi(x_a, y_a)

    # compute the flux across the line segment
    flux = fi.get_flux(edge_values)
    print(f'flux = {flux}')

    exact_flux = phi(p1[0], p1[1]) - phi(p0[0], p0[1])
    print(f'exact_flux = {exact_flux}')
    assert abs(flux - exact_flux) < 1.e-10


def test_river_big():

    # boundary points
    xyb = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.6), (3.0, 1.0), (5.0, 1.0), (5.0, 2.0), (4.0, 2.0), (2.0, 3.0), (0.0, 0.5),]
    n = len(xyb)
    # markers for boundary points, not used in this test but required by Triangle
    markers = [1 for _ in range(n)]
    # segments connecting the boundary points, must close the loop
    segs = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]

    # triangluate the domain, mode 'pzq10eQ' means:
    # p: triangulate a Planar Straight Line Graph
    # z: number all items starting from zero
    # q10: quality mesh with minimum angle of 10 degrees
    # e: output edge list
    # Q: quiet mode, no terminal output
    tri = Triangle()
    tri.set_points(xyb, markers=markers)
    tri.set_segments(segs)
    tri.triangulate(area=0.0001, mode='pzq10eQ')

    points = np.asarray([xy[0] for xy in tri.get_points()])
    triangles = np.asarray([t[0] for t in tri.get_triangles()])
    edges = np.asarray([e[0] for e in tri.get_edges()]) 

    # create a flux integrator for the line segment
    p0 = (0.0, 0.0)
    p1 = (2.0, 3.0)
    t0 = time.time()
    fi = FluxIntegrator(points, triangles, edges, p0=p0, p1=p1)
    t1 = time.time()
    print(f'FluxIntegrator computation of the weights took {t1 - t0:.4f} seconds for {len(triangles)} triangles and {len(edges)} edges')

    # print(f'points = {points}')
    # print(f'triangles = {triangles}')
    # plt.figure()
    # plt.triplot(points[:, 0], points[:, 1], triangles)
    # plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=2)
    # plt.show()

    # set flux values on edge based on a potential field phi = y
    def phi(x, y):
        return y
    
    edge_values = np.zeros((edges.shape[0],), float)
    for iaib in edges:
        ia, ib = iaib
        x_a, y_a = points[ia]
        x_b, y_b = points[ib]
        edge_id = fi.nodes_edge[tuple(iaib)]
        edge_values[edge_id] = phi(x_b, y_b) - phi(x_a, y_a)

    # compute the flux across the line segment
    t0 = time.time()
    flux = fi.get_flux(edge_values)
    t1 = time.time()
    print(f'flux = {flux} computed in {t1 - t0:.4f} seconds')

    exact_flux = phi(p1[0], p1[1]) - phi(p0[0], p0[1])
    print(f'exact_flux = {exact_flux}')
    assert abs(flux - exact_flux) < 1.e-10


def test_loop():

    # boundary points
    xyb = [(0.0, 0.0), (1.0, 0.2), (2.0, 0.6), (3.0, 1.0), (5.0, 1.0), (5.0, 2.0), (4.0, 2.0), (2.0, 3.0), (0.0, 0.5),]
    n = len(xyb)
    # markers for boundary points, not used in this test but required by Triangle
    markers = [1 for _ in range(n)]
    # segments connecting the boundary points, must close the loop
    segs = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]

    # triangluate the domain, mode 'pzq10eQ' means:
    # p: triangulate a Planar Straight Line Graph
    # z: number all items starting from zero
    # q10: quality mesh with minimum angle of 10 degrees
    # e: output edge list
    # Q: quiet mode, no terminal output
    tri = Triangle()
    tri.set_points(xyb, markers=markers)
    tri.set_segments(segs)
    tri.triangulate(area=0.5, mode='pzq10eQ')

    points = np.asarray([xy[0] for xy in tri.get_points()])
    triangles = np.asarray([t[0] for t in tri.get_triangles()])
    edges = np.asarray([e[0] for e in tri.get_edges()]) 

    # create a flux integrator for the line segment
    p0 = (0.0, 0.0)
    p1 = (2.0, 1.0)
    p2 = (1.0, 1.0)
    fi0 = FluxIntegrator(points, triangles, edges, p0=p0, p1=p1)
    fi1 = FluxIntegrator(points, triangles, edges, p0=p1, p1=p2)
    fi2 = FluxIntegrator(points, triangles, edges, p0=p2, p1=p0)

    # print(f'points = {points}')
    # print(f'triangles = {triangles}')
    # plt.figure()
    # plt.triplot(points[:, 0], points[:, 1], triangles)
    # plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=2)
    # plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
    # plt.plot([p2[0], p0[0]], [p2[1], p0[1]], 'r-', linewidth=2)
    # plt.show()

    # set flux values on edge based on a potential field phi = y
    def phi(x, y):
        return y
    
    edge_values = np.zeros((edges.shape[0],), float)
    for iaib in edges:
        ia, ib = iaib
        x_a, y_a = points[ia]
        x_b, y_b = points[ib]
        edge_id = fi0.nodes_edge[tuple(iaib)] # any fi will do since all the flux integrators share the same grid
        edge_values[edge_id] = phi(x_b, y_b) - phi(x_a, y_a)

    # compute the flux across the line segment
    flux = fi0.get_flux(edge_values) + fi1.get_flux(edge_values) + fi2.get_flux(edge_values)
    print(f'flux = {flux}')

    exact_flux = 0.0 # since we are integrating around a closed loop, the flux should be zero
    print(f'exact_flux = {exact_flux}')
    assert abs(flux - exact_flux) < 1.e-10


def test_aligned():

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

    # extract points, get_points returns a list of tuples (point, marker)
    points = np.asarray([xy[0] for xy in tri.get_points()])
    # extract triangles, get_triangles returns a list of tuples (triangle, marker)
    triangles = np.asarray([t[0] for t in tri.get_triangles()])
    # extract edges, get_edges returns a list of tuples (edge, marker)
    edges = np.asarray([e[0] for e in tri.get_edges()])

    print(f'points = {points}')
    print(f'triangles = {triangles}')
    print(f'edges = {edges}')

    p0 = (0.0, 0.0)
    p1 = (0.0, 1.0)
    fi = FluxIntegrator(points, triangles, edges, p0=p0, p1=p1)

    # set flux values on edge based on a potential field phi = y
    def phi(x, y):
        return y

    edge_values = np.zeros((edges.shape[0],), float)
    for iaib in edges:
        ia, ib = iaib
        x_a, y_a = points[ia]
        x_b, y_b = points[ib]
        edge_id = fi.nodes_edge[tuple(iaib)]
        # edge values are the difference in potential across the edge, phi_b - phi_a
        edge_values[edge_id] = phi(x_b, y_b) - phi(x_a, y_a)

    # compute the flux across the line segment
    flux = fi.get_flux(edge_values)
    print(f'flux = {flux}')

    exact_flux = phi(p1[0], p1[1]) - phi(p0[0], p0[1])
    assert abs(flux - exact_flux) < 1.e-10
