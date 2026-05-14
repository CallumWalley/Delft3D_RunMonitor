import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree


class VolumeIntegrator:
    """
    Precompute geometric intersections once,
    then integrate rapidly for many timesteps.
    """

    def __init__(
        self,
        points,
        triangles,
        clip_polygon_coords,
    ):

        self.points = points
        self.triangles = triangles

        clip_poly = Polygon(clip_polygon_coords)

        #
        # Build triangle polygons
        #
        tri_polys = []

        for tri in triangles:
            tri_polys.append(
                Polygon(points[tri])
            )

        #
        # Spatial index
        #
        tree = STRtree(tri_polys)

        #
        # Candidate cells
        #
        candidate_ids = tree.query(clip_poly)

        #
        # Precompute conservative weights
        #
        cell_ids = []
        weights = []


        for icell in candidate_ids:

            poly = tri_polys[icell]

            inter = poly.intersection(clip_poly)

            if inter.is_empty:
                continue

            area = inter.area

            if area <= 0.0:
                continue

            cell_ids.append(icell)
            weights.append(area)

        self.cell_ids = np.asarray(cell_ids, dtype=np.int64)
        self.weights = np.asarray(weights)

    def get_volume(self, height):
        """
        Compute volume for one timestep.

        Parameters
        ----------
        height : (ncells,) array
            Cell-centered heights.

        Returns
        -------
        volume : float
        """

        return np.sum(
            height[self.cell_ids] * self.weights
        )