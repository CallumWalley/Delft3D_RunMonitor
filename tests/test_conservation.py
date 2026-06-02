import numpy as np
import matplotlib.pyplot as plt
from triangle import Triangle
import time

from Delft3D_RunMonitor import FluxIntegrator, VolumeIntegrator

def test_straight_duct():

    # Create a simple straight duct mesh
    bound_points = [[0., 0.], [10., 0.], [10., 1.], [0., 1.]]
    bound_segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
    tri = Triangle()
    tri.set_points(bound_points)
    tri.set_segments(bound_segments)
    tri.triangulate(area=0.1, mode='pzq20eQ')
    points = np.asarray([xy[0] for xy in tri.get_points()])
    triangles = np.asarray([t[0] for t in tri.get_triangles()])
    edges = np.asarray([e[0] for e in tri.get_edges()]) 

    p0 = (0., 0.0)
    p1 = (0., 1.0)
    p2 = (10., 0.0)
    p3 = (10., 1.0)
    fi01 = FluxIntegrator(points, triangles, edges, p0=p0, p1=p1)
    fi23 = FluxIntegrator(points, triangles, edges, p0=p2, p1=p3)

    # set flux values on edge based on a potential field phi = y
    def phi(x, y):
        return y

    edge_fluxes = np.empty(len(edges), float)
    for i, (n1, n2) in enumerate(edges):
        x1, y1 = points[n1]
        x2, y2 = points[n2]
        edge_fluxes[i] = phi(x2, y2) - phi(x1, y1)

    # plt.figure()
    # plt.triplot(points[:, 0], points[:, 1], triangles)
    # plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=2)
    # plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'r-', linewidth=2)
    # plt.axis('equal')
    # plt.show()

    # Compute the fluxes across the line segments
    flux_01 = fi01.get_flux(edge_fluxes)
    flux_23 = fi23.get_flux(edge_fluxes)

    # because the flow derives from a stream function, the flux across the line segments should be exactly 1.0, and the same for both segments
    # since the endpoints are nodes of the mesh.
    assert abs(flux_01 - 1.0) < 1e-10, f"Flux across the line segment p0-p1 should be 1.0, but got {flux_01}"
    assert abs(flux_23 - 1.0) < 1e-10, f"Flux across the line segment p2-p3 should be 1.0, but got {flux_23}"
    assert abs(flux_23 - flux_01) < 1e-10, f"Fluxes across the two line segments should be equal, but got {flux_01} and {flux_23}"

def test_twisted_duct():

    # Create a simple straight duct mesh
    bound_points = [[0., 0.], [10., 0.], [10., 1.], [0., 1.]]
    bound_segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
    tri = Triangle()
    tri.set_points(bound_points)
    tri.set_segments(bound_segments)
    tri.triangulate(area=0.1, mode='pzq20eQ')
    points = np.asarray([xy[0] for xy in tri.get_points()])
    triangles = np.asarray([t[0] for t in tri.get_triangles()])
    edges = np.asarray([e[0] for e in tri.get_edges()]) 

    p0 = (0., 0.0)
    p1 = (0., 1.0)
    p2 = (10., 0.0)
    p3 = (10., 1.0)

    # set flux values on edge based on a potential field phi = y
    def phi(x, y):
        return y

    edge_fluxes = np.empty(len(edges), float)
    for i, (n1, n2) in enumerate(edges):
        x1, y1 = points[n1]
        x2, y2 = points[n2]
        edge_fluxes[i] = phi(x2, y2) - phi(x1, y1)

    # now deform the mesh by perturbning it with a sin function. Note that
    # the stream function values do not change (they are scalars). Therefore,
    # the flow is invariant under the deformation, and the flux across the line 
    # segments should be the same as before, even though the geometry is different.
    def deform(x, y):
        twist_angle = np.pi
        xprime = x 
        yprime = y + 4.0 * np.sin(twist_angle*x/10)
        return xprime, yprime

    points_prime = np.empty_like(points)
    for i in range(len(points)):
        x, y = points[i]
        xprime, yprime = deform(x, y)
        points_prime[i, :] = [xprime, yprime]

    p0prime = deform(*p0)
    p1prime = deform(*p1)
    p2prime = deform(*p2)
    p3prime = deform(*p3)

    # plt.figure()
    # plt.triplot(points_prime[:, 0], points_prime[:, 1], triangles)
    # plt.plot([p0prime[0], p1prime[0]], [p0prime[1], p1prime[1]], 'r-', linewidth=2)
    # plt.plot([p2prime[0], p3prime[0]], [p2prime[1], p3prime[1]], 'r-', linewidth=2)
    # plt.axis('equal')
    # plt.show()

    # Compute the fluxes across the line segments
    fi01 = FluxIntegrator(points_prime, triangles, edges, p0=p0prime, p1=p1prime)
    fi23 = FluxIntegrator(points_prime, triangles, edges, p0=p2prime, p1=p3prime)
    flux_01 = fi01.get_flux(edge_fluxes)
    flux_23 = fi23.get_flux(edge_fluxes)

    assert abs(flux_23 - flux_01) < 1e-10, f"Fluxes across the two line segments should be equal, but got {flux_01} and {flux_23}"
