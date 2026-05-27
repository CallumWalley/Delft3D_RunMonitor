import numpy as np


class FluxIntegrator:
    """
    Compute the intersections of a 2D line segment with a triangular mesh.

    Parameters
    ----------
    points : (N, 2) array
        Node coordinates.
    triangles : (M, 3) array
        Triangle connectivity.
    p0 : (2,) array-like
        Start point of the line.
    p1 : (2,) array-like
        End point of the line.

    Attributes
    ----------
    segments : list of tuples
        Each tuple is

            (triangle_id, xi_a, eta_a, xi_b, eta_b)

        where (xi_a, eta_a) and (xi_b, eta_b) are the barycentric
        coordinates of the segment endpoints inside the triangle.
    """

    def __init__(self, points, triangles, edges, p0, p1, tol=1e-12):

        self.points = np.asarray(points, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        self.edges = np.asarray(edges, dtype=int)

        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)

        self.tol = tol

        self._build_nodes_edge()
        self.segments = self._build_segments()

    @staticmethod
    def _cross2(a, b):
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
        return np.all(bary >= -self.tol)
    
    def _build_nodes_edge(self):

        self.nodes_edge = {tuple(iaib) : ie for ie, iaib in enumerate(self.edges)}

    def _build_segments(self):

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

                if (ia, ib) in self.nodes_edge:
                    edge_id = self.nodes_edge[(ia, ib)]
                    sign = 1
                elif (ib, ia) in self.nodes_edge:
                    edge_id = self.nodes_edge[(ib, ia)]
                    sign = -1
                else:
                    raise RuntimeError(f"Cannot find edge {ia} -> {ib}")
            
                self.weights[edge_id] = self.weights.get(edge_id, 0) + weight * sign


    def get_flux(self, edge_values):
        """
        Compute the flux across the line segment by summing the contributions from the intersected edges.

        Parameters
        ----------
        edge_values : dict
            A dictionary mapping edge_id to the value of the field on that edge.

        Returns
        -------
        flux : float
            The computed flux across the line segment.
        """

        flux = 0.0

        for edge_id, weight in self.weights.items():
            value = edge_values[edge_id]
            flux += weight * value

        return flux

