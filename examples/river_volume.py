import matplotlib.pylab as plt
import numpy as np
from glob import glob
import defopt
from Delft3D_RunMonitor import MultiUGridMesh, VolumeIntegrator
import matplotlib.pylab as plt

def main(*, mappattern: str='FlowFM_*_map.nc', time_index: int=1, 
            x0b: float=2056400., y0b: float=5785690., x0e: float=2056600., y0e: float=5787000.,
            x1b: float=2057200., y1b: float=5785690., x1e: float=2057000., y1e: float=5787000.,
            show_plot: bool=False):
    """
    Compute the water volume between two segments

    Args:
        mappattern: Glob pattern for map files, or a single filename
        time_index: Time index
        x0b: First segment start x coordinate
        y0b: First segment start y coordinate
        x0e: First segment end x coordinate
        y0e: First segment end y coordinate
        x1b: Second segment start x coordinate
        y1b: Second segment start y coordinate
        x1e: Second segment end x coordinate
        y1e: Second segment end y coordinate
        show_plot: Whether to show a plot of the river geometry and intersection polygon
     """
    #
    # Build the river geometry
    #
    ugrid = MultiUGridMesh(sorted(glob(mappattern)))

    #
    # Build the intersection polygon (counterclockwise direction)
    #
    poly = np.array(
        [
         (x1b, y1b),
         (x1e, y1e),
         (x0e, y0e),
         (x0b, y0b),
        ]
    )

    if show_plot:
         plt.figure()
         plt.plot(poly[:, 0], poly[:, 1], 'r-')
         plt.plot([poly[-1, 0], poly[0, 0]], [poly[-1, 1], poly[0, 1]], 'r-')
         plt.axis('equal')
         plt.title('Intersection polygon')
    #
    # Compute the volume of each subdomain intersection with the polygon
    #
    volume_total = 0.0
    for mesh in ugrid.meshes:
        points = np.column_stack((mesh.x, mesh.y))
        vi = VolumeIntegrator(points, mesh.face_nodes, poly)
        # read the data
        depth = mesh.readField(varname='mesh2d_waterdepth', time_index=time_index)
        # compute the volume 
        volume_total += vi.get_volume(depth)

        if show_plot:
            plt.triplot(points[:, 0], points[:, 1], mesh.face_nodes)

    print(f'Total volume: {volume_total:.0f} m^3 at time index {time_index}')

    if show_plot:
        plt.title(f'Total volume: {volume_total:.0f} m^3')
        plt.show()



if __name__ == '__main__':
    defopt.run(main)