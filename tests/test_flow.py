import numpy as np
import matplotlib.pyplot as plt
import time

from Delft3D_RunMonitor import FluxIntegrator, UGridMesh
 
def test_conservation():

    mesh = UGridMesh('data/FlowFM_0001_map.nc')
    points = np.column_stack((mesh.x, mesh.y))

    # last time (must be close to steady state)
    time_index = len(mesh.time) - 1

    num_faces = mesh.face_nodes.shape[0]
    flows = []

    # itereate over a few triangles
    for face_id in (1000, 5000, 10000, 20000,):

        # get the node ids of the triangle
        point_ids = mesh.face_nodes[face_id]

        # get the node coordinates of the triangle, these form the polyline
        # across which we comopute the flow
        face_points = points[point_ids]

        # slightly offset the points towards the centre to avoid the intersects to become
        # colinear with the triangle edges, otherwise fluxes are doubly 
        # counted.
        mid_point = face_points.mean(axis=0)
        eps = 1.235446e-8
        face_points += (mid_point - face_points) * eps

        # flow across the three edges
        fi01 = FluxIntegrator(points, mesh.face_nodes, mesh.edge_nodes, 
                          face_points[0,:2], face_points[1,:2])
        fi12 = FluxIntegrator(points, mesh.face_nodes, mesh.edge_nodes, 
                          face_points[1,:2], face_points[2,:2])
        fi20 = FluxIntegrator(points, mesh.face_nodes, mesh.edge_nodes, 
                          face_points[2,:2], face_points[0,:2])
        
        # extract the lateral area integrated flow at the current time
        q1 = mesh.readField(varname='mesh2d_q1', time_index=time_index)

        flowQ01 = fi01.get_flux(q1)
        flowQ12 = fi12.get_flux(q1)
        flowQ20 = fi20.get_flux(q1)

        # collect the result
        flows.append(flowQ01 + flowQ12 + flowQ20)

    # expect nearly zero flux since closed loop
    assert max(np.fabs(flows)) < 1.e-5

