from typing import List
import matplotlib.pylab as plt
import numpy as np
from typing import List
import defopt
from Delft3D_RunMonitor import MultiUGridMesh, FluxIntegrator
import matplotlib.pylab as plt

def main(*, mapnames: List[str]=['FlowFM_0000_map.nc'], time_index: int=1, 
            x0b: float=2056400., y0b: float=5785690., x0e: float=2056600., y0e: float=5787000.,
            show_plot: bool=False):
    """
    Compute the flow across a river cross-section defined by a line segment.

    Args:
        mapnames: List of map filenames (e.g. ['FlowFM_0000_map.nc', 'FlowFM_0001_map.nc'])
        time_index: Time index
        x0b: First segment start x coordinate
        y0b: First segment start y coordinate
        x0e: First segment end x coordinate
        y0e: First segment end y coordinate
        show_plot: Whether to show a plot of the river geometry and intersection polygon
     """
    #
    # Build the river geometry
    #
    print(mapnames)
    ugrid = MultiUGridMesh(sorted(mapnames))

    if show_plot:
        plt.figure()
        plt.plot([x0b, x0e], [y0b, y0e], 'r-')

    flow_total = 0.0
    for mesh in ugrid.meshes:

        print(f'Processing mesh: {mesh.nc.filepath()}')
        print(f'  Number of points: {len(mesh.x)} faces: {len(mesh.face_nodes)} edges: {len(mesh.edge_nodes)}')

        points = np.column_stack((mesh.x, mesh.y))

        if show_plot:
            plt.triplot(points[:, 0], points[:, 1], mesh.face_nodes)

        #
        # Read the data
        #

        # edge centred integrated flow values at the current time step
        q1 = mesh.readField(varname='mesh2d_q1', time_index=time_index)

        # compute the flow 
        fi = FluxIntegrator(points, mesh.face_nodes, mesh.edge_nodes, (x0b, y0b), (x0e, y0e))
        flux_increment = fi.get_flux(edge_values=q1)

        print(f'  Flow increment: {flux_increment:.0f} m^3/s')
        flow_total += flux_increment

    print(f'Total flow: {flow_total:.0f} m^3/s at time index {time_index}')

    if show_plot:
        plt.title(f'Total flow: {flow_total:.0f} m^3/s at time index {time_index}')
        plt.show()



if __name__ == '__main__':
    defopt.run(main)