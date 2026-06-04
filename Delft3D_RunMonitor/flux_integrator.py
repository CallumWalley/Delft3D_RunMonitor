import numpy as np


class FluxIntegrator:
    """
    Compute the lateral flux across a 2D line intersecting triangular mesh.
    """

    def __init__(self, points, triangles, edges, p0, p1, tol=1e-12):
        """
        Constructor
         - points: Nx2 array of node coordinates
         - triangles: Mx3 array of triangle vertex indices (0-based)
         - edges: Kx2 array of edge vertex indices (0-based)
         - p0, p1: endpoints of the line segment (2D coordinates)
         - tol: tolerance for geometric computations
        """

        self.points = np.asarray(points, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        self.edges = np.asarray(edges, dtype=int)

        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)

        self.tol = tol

        # node (ia, ib) to edge (ie) connectivity
        self.nodes_edge = {tuple(iaib) : ie for ie, iaib in enumerate(self.edges)}

        # copute the interpolation weights. This depends on the geometry only, 
        # not on the field values, so we can compute it once and reuse for
        # time steps.
        self.segments = self._compute_weights()

        # compute edge lengths
        self.edge_lengths = np.linalg.norm( \
                            self.points[self.edges[:, 1]] - self.points[self.edges[:, 0]], \
                                axis=1)

    @staticmethod
    def _cross2(a, b):
        # 2D cross product (scalar), returns the z-component of the 3D cross product
        return a[0] * b[1] - a[1] * b[0]

    def _line_segment_intersection_parameter(self, a, b):
        """
        Intersect the global line segment p(t)=p0+t*(p1-p0), t in [0,1]
        with edge segment [a,b].

        Returns
        -------
        t : float or None
            Parameter on the global line if an intersection exists.
        """

        p = self.p0
        r = self.p1 - self.p0

        q = a
        s = b - a

        rxs = self._cross2(r, s)
        qmp = q - p

        if abs(rxs) < self.tol:
            # WHAT SHOULD WE DO IF r s overlap? For now, we just ignore this case.
            return None  # Parallel

        t = self._cross2(qmp, s) / rxs
        u = self._cross2(qmp, r) / rxs

        if -self.tol <= t <= 1 + self.tol and -self.tol <= u <= 1 + self.tol:
            return np.clip(t, 0.0, 1.0)

        return None

    def _point_in_triangle_barycentric(self, p, tri_pts):
        """
        Return barycentric coordinates (l0,l1,l2).
        """

        a, b, c = tri_pts

        v0 = b - a
        v1 = c - a
        v2 = p - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01

        if abs(denom) < self.tol:
            return None

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        return np.array([u, v, w])

    def _inside_triangle(self, bary):
        # bary are the full barycentric coordinates (l0, l1, l2), l0 + l1 + l2 = 1
        return np.all(bary >= -self.tol)
    
    def _compute_weights(self):

        segments = []

        line_dir = self.p1 - self.p0

        for tri_id, tri in enumerate(self.triangles):

            tri_pts = self.points[tri]

            #
            # Collect intersection parameters along the global line
            #
            tvals = []

            #
            # Add endpoints if they are inside the triangle
            #
            for t in [0.0, 1.0]:
                p = self.p0 + t * line_dir
                bary = self._point_in_triangle_barycentric(p, tri_pts)

                if bary is not None and self._inside_triangle(bary):
                    tvals.append(t)

            #
            # Intersections with triangle edges
            #
            edges = [(0, 1), (1, 2), (2, 0)]

            for i, j in edges:

                t = self._line_segment_intersection_parameter(
                    tri_pts[i], tri_pts[j]
                )

                if t is not None:
                    tvals.append(t)

            #
            # Remove duplicates
            #
            if not tvals:
                continue

            tvals = np.array(sorted(tvals))

            unique_t = [tvals[0]]

            for t in tvals[1:]:
                if abs(t - unique_t[-1]) > self.tol:
                    unique_t.append(t)

            #
            # Need exactly two distinct points to define
            # the segment inside the triangle
            #
            if len(unique_t) < 2:
                continue

            ta = unique_t[0]
            tb = unique_t[-1]

            if tb - ta < self.tol:
                continue

            pa = self.p0 + ta * line_dir
            pb = self.p0 + tb * line_dir

            bary_a = self._point_in_triangle_barycentric(pa, tri_pts)
            bary_b = self._point_in_triangle_barycentric(pb, tri_pts)

            if bary_a is None or bary_b is None:
                continue

            #
            # Return only xi, eta.
            # Third barycentric coord = 1 - xi - eta
            #
            xi_a, eta_a = bary_a[1], bary_a[2]
            xi_b, eta_b = bary_b[1], bary_b[2]

            segments.append(
                (
                    tri_id,
                    xi_a,
                    eta_a,
                    xi_b,
                    eta_b,
                )
            )

        self.weights = {}

        for face_id, xi_a, eta_a, xi_b, eta_b in segments:

            xibar = 0.5*(xi_a + xi_b)
            etabar = 0.5*(eta_a + eta_b)
            dxi = xi_b - xi_a
            deta = eta_b - eta_a
            one_minus_sum = 1.0 - xibar - etabar

            tri = self.triangles[face_id]

            # weights for the three edges of the triangle, in the order (0,1), (1,2), (2,0)
            ws = ( \
                one_minus_sum*dxi + xibar*(dxi + deta), \
                xibar*deta - etabar*dxi, \
                -etabar*(dxi + deta) - one_minus_sum*deta, \
            )

            for i in range(3):

                weight = ws[i]
                ia, ib = tri[i], tri[(i + 1) % 3]
                edge_id = -1

                if (ia, ib) in self.nodes_edge:
                    edge_id = self.nodes_edge[(ia, ib)]
                    sign = 1
                elif (ib, ia) in self.nodes_edge:
                    edge_id = self.nodes_edge[(ib, ia)]
                    sign = -1
                else:
                    print(f"ERROR: Cannot find edge {edge_id} {ia} -> {ib} or {ib} -> {ia} in the edge list")
                    print(f"  Triangle {face_id} has nodes {tri} with coordinates {self.points[tri]}")
                    raise RuntimeError(f"Cannot find edge {ia} -> {ib}")
            
                self.weights[edge_id] = self.weights.get(edge_id, 0) + weight * sign


    def get_flux(self, edge_values: np.ndarray) -> float:
        """
        Compute the flux across the line segment by summing the contributions from the intersected edges.

        Parameters
        ----------
        edge_values : Array-like
            Value of the field on each edge. This should be flux integrated value for each edge.
            If using point values at the edge midpoints, then the edge_value should be the point value
            multiplied by the edge length and vertical depth. 
            
            Beware of the sign convention for the flux values on the edges, the flux is in the 
            direction of (xb - xa, yb - ya, 0) x (0, 0, 1)  where (xa, ya) and (xb, yb) are the 
            2D coordinates of the edge vertices.

        Returns
        -------
        flux : float
            The computed flux across the line segment.
        """

        flux = 0.0

        # sum up the values on the edges, weighted by the precomputed weights
        for edge_id, weight in self.weights.items():
            value = edge_values[edge_id]
            flux += weight * value

        return flux
    

    def get_flow_from_u(self, u: np.ndarray, depths: np.ndarray, edge_faces: np.ndarray) -> float:
        """
        Compute the flow across the line segment from the edge velocity values. 
        This is a convenience method that combines the edge velocity with the edge lengths and 
        depths to compute the flux.

        Parameters
        ----------
        u : Array-like of shape (E,)
            Velocity value on each edge (point value at the edge midpoint).
        depths : Array-like of shape (K,)
            Water depths for each triangle.
        edge_faces : Array-like of shape (E, 2)
            For each edge, the indices of the two adjacent triangles (or -1 for boundary edges).

        Returns
        -------
        flow : float
            The computed flow across the line segment.
        """

        # compute the flow across each edge by multiplying the velocity with the edge length and 
        # the average depth of the adjacent triangles
        flow_values = np.zeros(len(self.edges))
        for i, (face_a, face_b) in enumerate(edge_faces):

            # Compute the lateral area associated with this edge, 
            # typically the edge length multiplied by the average depth of the 
            # two adjacent triangles. For boundary edges, we can use the depth of the 
            # single adjacent triangle.
            lateral_area = 0.0
            mid_point_face_a = None
            mid_point_face_b = None
            sign = 0.0
            if face_a >= 0 and face_b >= 0:
                lateral_area = 0.5 * (depths[face_a] + depths[face_b]) * self.edge_lengths[i]
                mid_point_face_a = self.points[self.triangles[face_a]].mean(axis=0)
                mid_point_face_b = self.points[self.triangles[face_b]].mean(axis=0)
            elif face_a >= 0:
                lateral_area = depths[face_a] * self.edge_lengths[i]
                mid_point_face_a = self.points[self.triangles[face_a]].mean(axis=0)
                # use the mid edge point as the mid point for the boundary edge, since we only have one adjacent triangle
                mid_point_face_b = self.points[self.edges[i]].mean(axis=0)
            elif face_b >= 0:
                lateral_area = depths[face_b] * self.edge_lengths[i]
                mid_point_face_b = self.points[self.triangles[face_b]].mean(axis=0)
                # use the mid edge point as the mid point for the boundary edge, since we only have one adjacent triangle
                mid_point_face_a = self.points[self.edges[i]].mean(axis=0)

            # The convention in Delft3D is that the flow is positive in the direction of left to 
            #` right face.
            sign = 1.0 if self._cross2(mid_point_face_b - mid_point_face_a, self.p1 - self.p0) > 0 else -1.0

            flow_values[i] = sign * lateral_area * u[i]

        # Now compute the flow from the edge integrated flux values...
        return self.get_flux(flow_values)