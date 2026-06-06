import matplotlib.pylab as plt
import numpy as np
from typing import List
import defopt
from Delft3D_RunMonitor import UGridMesh, FluxIntegrator
import matplotlib.pylab as plt

def main(*, mapname: str='FlowFM_0000_map.nc', 
            face_index: int=4010,
            show_plot: bool=True):
    """
    Check conservation of flow across a triangle

    Args:
        mapname: Map filename (e.g. 'FlowFM_0000_map.nc')
        face_index: Index of the face (triangle) to check
        show_plot: Whether to show a plot
     """
    #
    # Build the river geometry
    #
    print(mapname)
    mesh = UGridMesh(mapname)

    print(f'  Number of points: {len(mesh.x)} faces: {len(mesh.face_nodes)} edges: {len(mesh.edge_nodes)}')

    points = np.column_stack((mesh.x, mesh.y))
    point_ids = mesh.face_nodes[face_index]
    face_points = points[point_ids]

    # slightly offset the points towards the centre to avoid the intersects to become
    # colinear with the triangle edges. This prevents the problem where fluxes a doubly 
    # counted.
    mid_point = face_points.mean(axis=0)
    eps = 1.235446e-8
    face_points += (mid_point - face_points) * eps


    fi01 = FluxIntegrator(points, mesh.face_nodes, mesh.edge_nodes, 
                          face_points[0,:2], face_points[1,:2])
    fi12 = FluxIntegrator(points, mesh.face_nodes, mesh.edge_nodes, 
                          face_points[1,:2], face_points[2,:2])
    fi20 = FluxIntegrator(points, mesh.face_nodes, mesh.edge_nodes, 
                          face_points[2,:2], face_points[0,:2])
    
    print(f'face: {face_index} point_ids={point_ids} face_points={face_points}')
    print(f'edge 0->1: {face_points[0,:2]} -> {face_points[1,:2]}')
    print(f'edge 1->2: {face_points[1,:2]} -> {face_points[2,:2]}')
    print(f'edge 2->0: {face_points[2,:2]} -> {face_points[0,:2]}')

    print(f'fi01.weights = {fi01.weights}')
    print(f'fi12.weights = {fi12.weights}')
    print(f'fi20.weights = {fi20.weights}')
    
    flows_u = []
    flows_q = []

    for time_index, tm in enumerate(mesh.time):

        #
        # Read the data
        #

        # edge centred velocity values at the current time step
        u1 = mesh.readField(varname='mesh2d_u1', time_index=time_index)
        q1 = mesh.readField(varname='mesh2d_q1', time_index=time_index)

        # face centred depth values
        depth = mesh.readField(varname='mesh2d_waterdepth', time_index=time_index)

        #
        # total flow across the triangle, the edges are such that the flows point outwards
        #

        # flow from the u velocity, need to multiply with surface element
        flowU01 = fi01.get_flow_from_u(u=u1, depths=depth, edge_faces=mesh.edge_faces)
        flowU12 = fi12.get_flow_from_u(u=u1, depths=depth, edge_faces=mesh.edge_faces)
        flowU20 = fi20.get_flow_from_u(u=u1, depths=depth, edge_faces=mesh.edge_faces)
        flowU = flowU01 + flowU12 + flowU20
        flows_u.append(flowU)

        # flow from q, already integrated across the element
        flowQ01 = fi01.get_flux(q1)
        flowQ12 = fi12.get_flux(q1)
        flowQ20 = fi20.get_flux(q1)
        flowQ = flowQ01 + flowQ12 + flowQ20
        flows_q.append(flowQ)

        print(f'  time index: {time_index:>6}  time: {tm:>10.2f} s total flow U/Q = {flowU:>12.3e}/{flowQ:>12.3e} edge contributions: {flowU01:.2f}/{flowQ01:.2f} {flowU12:.2f}/{flowQ12:.2f} {flowU20:.2f}/{flowQ20:.2f} m^3/s')


    if show_plot:
        plt.figure()
        #plt.plot(mesh.time[:], flows_u, 'b-') # need to fix a sign error
        plt.plot(mesh.time[:], flows_q, 'm-')
        plt.xlabel(f'time {mesh.time.units}')
        plt.ylabel('flow (m^3/s)')
        plt.title(f'Outward flow from triangle {face_index} in {mapname}')


        # plot the face and the triangular mesh
        _, ax = plt.subplots()

        # plot the full triangular mesh in black
        ax.triplot(mesh.x, mesh.y, mesh.face_nodes[:, :3], color='k', linewidth=0.5)

        # highlight the selected triangle in red
        hi = mesh.face_nodes[face_index, :3]
        ax.fill(mesh.x[hi], mesh.y[hi], color='red', alpha=0.3)
        ax.plot(mesh.x[hi[[0, 1, 2, 0]]], mesh.y[hi[[0, 1, 2, 0]]], 'r-', linewidth=2)

        ax.set_aspect('equal')
        ax.set_title(f'Triangle {face_index} in {mapname}')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        plt.show()



if __name__ == '__main__':
    defopt.run(main)