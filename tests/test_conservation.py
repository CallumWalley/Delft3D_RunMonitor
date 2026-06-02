import numpy as np
import matplotlib.pyplot as plt
from triangle import Triangle
import time

from Delft3D_RunMonitor import FluxIntegrator, VolumeIntegrator

def test_straight_duct():

    # Create a simple straight duct mesh
    bound_points = [[0., 0.], [10., 0.], [1., 1.], [0., 1.]]
    bound_segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
    tri = Triangle()
    tri.set_points(bound_points)
    tri.set_segments(bound_segments)
    tri.triangulate(area=0.5, mode='pzq10eQ')
    points = np.asarray([xy[0] for xy in tri.get_points()])
    triangles = np.asarray([t[0] for t in tri.get_triangles()])
    edges = np.asarray([e[0] for e in tri.get_edges()]) 

    p0 = (0.5, 0.0)
    p1 = (0.5, 1.0)
    p2 = (9.5, 0.0)
    p3 = (9.5, 1.0)
    fi01 = FluxIntegrator(points, triangles, edges, p0=p0, p1=p1)
    fi23 = FluxIntegrator(points, triangles, edges, p0=p2, p1=p3)

    # set flux values on edge based on a potential field phi = y
    def phi(x, y):
        return y

    edge_fluxes = np.array([phi(*points[edge].mean(axis=0)) for edge in edges])

    # Compute the fluxes across the line segments
    flux_01 = fi01.get_flux(edge_fluxes)
    flux_23 = fi23.get_flux(edge_fluxes)

    assert abs(flux_23 - (-flux_01)) < 1e-10, f"Fluxes across the two line segments should be equal, but got {flux_01} and {flux_23}"
