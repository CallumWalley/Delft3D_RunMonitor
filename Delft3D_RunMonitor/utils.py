import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import splprep, splev
from sklearn.cluster import KMeans


import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import splprep, splev

def calculate_clean_centerline(points, num_clusters=20, num_output_points=100, stiffness=50.0):
    """
    Collapses a wide river mesh into a central thread using K-Means, 
    then fits a smooth, loop-free spline.
    """
    # 1. Use K-Means to find points directly in the middle of the river channel
    # This collapses the width of the river into a single centerline thread
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_centers = kmeans.fit(points).cluster_centers_
    
    # 2. Sort the cluster centers from one end of the river to the other
    # We find the leftmost cluster center to start the sequence
    start_idx = np.argmin(cluster_centers[:, 0])
    
    unvisited = list(range(num_clusters))
    unvisited.remove(start_idx)
    ordered_centers = [cluster_centers[start_idx]]
    current_center = cluster_centers[start_idx]
    
    while unvisited:
        # Find the nearest next cluster center
        distances = np.linalg.norm(cluster_centers[unvisited] - current_center, axis=1)
        nearest_idx = unvisited[np.argmin(distances)]
        
        ordered_centers.append(cluster_centers[nearest_idx])
        unvisited.remove(nearest_idx)
        current_center = cluster_centers[nearest_idx]
        
    ordered_centers = np.array(ordered_centers)
    
    # 3. Fit a stiff B-spline through the clean center thread
    # Because the points are already in the middle, loops are impossible
    pts = ordered_centers.T
    tck, u = splprep(pts, s=stiffness)
    
    # 4. Generate the final continuous centerline
    u_new = np.linspace(0, 1, num_output_points)
    centerline = splev(u_new, tck)
    
    return np.array(centerline).T
