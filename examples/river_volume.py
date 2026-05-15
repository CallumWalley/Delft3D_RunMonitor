import matplotlib.pylab as plt
import numpy as np
import triangle
import defopt
from Delft3D_RunMonitor import calculate_clean_centerline

def main(*, num_segs: int=10, nx: int=100):
    """
    Compute the volume river segments

    num_segs: number of river segments
    nx: number of x boundary river points
    """
    #
    # Build the river geometry
    #
    dx = 1.0
    xy_bottom = [ (i*dx, 20*np.sin(np.pi * i * dx/100.)) for i in range(nx)]
    xy_top = [ (i*dx, 5 + 30*np.sin(np.pi * i * dx/100.)) for i in range(nx)]
    xyb = xy_bottom + xy_top[::-1]
    n1 = len(xyb)
    markers = [1 for i in range(n1)]
    tri = triangle.Triangle()
    tri.set_points(xyb, markers=markers)
    bound_segs = [(i, i + 1) for i in range(n1 - 1)] + [(n1 - 1, 0)]
    tri.set_segments(bound_segs)
    tri.triangulate(area=1.0)

    xy = np.asarray([p[0] for p in tri.get_points()])
    triangles = np.asarray([t[0] for t in tri.get_triangles()])

    #
    # Find the center line
    #
    xyc = calculate_clean_centerline(xy, num_clusters=10, num_segments=num_segs, stiffness=50000.0)
    plt.figure()
    plt.triplot(xy[:, 0], xy[:, 1], triangles)
    plt.plot(xyc[:, 0], xyc[:, 1], 'r-')
    plt.show()


    #
    # Compute the volume of each section
    #


if __name__ == '__main__':
    defopt.run(main)