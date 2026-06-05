import matplotlib.pylab as plt
import numpy as np
from typing import List
import defopt
from Delft3D_RunMonitor import UGridMesh, FluxIntegrator
import matplotlib.pylab as plt

def main(*, mapname: str='FlowFM_0000_map.nc', 
            face_index: int=0,
            show_plot: bool=False):
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
    edge_nodes = mesh.face_nodes[face_index]
    edge_points = points[edge_nodes]

    fi01 = FluxIntegrator(points, mesh.face_nodes, mesh.edge_nodes, edge_points[0,:2], edge_points[1,:2])
    fi12 = FluxIntegrator(points, mesh.face_nodes, mesh.edge_nodes, edge_points[1,:2], edge_points[2,:2])
    fi20 = FluxIntegrator(points, mesh.face_nodes, mesh.edge_nodes, edge_points[2,:2], edge_points[0,:2])

    flows = []

    for time_index, tm in enumerate(mesh.time):

        #
        # Read the data
        #

        # edge centred velocity values at the current time step
        u1 = mesh.readField(varname='mesh2d_u1', time_index=time_index)
        # face centred depth values
        depth = mesh.readField(varname='mesh2d_waterdepth', time_index=time_index)

        # total flow across the triangle, the edges are such that the flows point outwards
        flow = fi01.get_flow_from_u(u=u1, depths=depth, edge_faces=mesh.edge_faces) \
                + fi12.get_flow_from_u(u=u1, depths=depth, edge_faces=mesh.edge_faces) \
                + fi20.get_flow_from_u(u=u1, depths=depth, edge_faces=mesh.edge_faces)
        
        flows.append(flow)

        print(f'  time index: {time_index:>6}  time: {tm:>10.2f} s flow = {flow:>12.3e} m^3/s')


    if show_plot:
        plt.figure()
        #plt.plot(range(len(mesh.time)), flows, 'o-')
        plt.plot(mesh.time[:], flows, 'o-')
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