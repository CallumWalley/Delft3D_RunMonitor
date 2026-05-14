import numpy as np
import matplotlib.pyplot as plt
import random

from Delft3D_RunMonitor import calculate_clean_centerline
 
def test_curved_river():

    # ---------------------------------------------------------
    # 1. synthetic curved centerline
    # ---------------------------------------------------------
    nt1 = 201
    radius = 100
    width = 20

    # parameter
    t = np.linspace(0.0, np.pi, nt1)

    # generate points
    points = np.empty((nt1, 2), float)
    random.seed(123)
    noise = np.array([random.random() - 0.5 for _ in range(nt1)])
    points[:, 0] = (radius + width * noise) * np.cos(t)
    points[:, 1] = (radius + width * noise) * np.sin(t)
 
    # ---------------------------------------------------------
    # 3. compute centerline
    # --------------------------------------------------------- 
    xyc = calculate_clean_centerline(points, num_clusters=20, num_output_points=100, stiffness=5000.0)
    print(f'xyc = {xyc}')
    plt.figure()
    triangles = []
    plt.scatter(points[:, 0], points[:, 1])
    plt.plot(xyc[:,0], xyc[:,1], 'r-')
    plt.gca().set_aspect('equal') # Keep aspect ratio correct
    plt.show()

    # ---------------------------------------------------------
    # 4. checks
    # ---------------------------------------------------------

    spread = np.abs(np.sqrt(xyc[:,0]**2 + xyc[:,1]**2) - radius)
    assert np.max(spread) < width
