import numpy as np
import pyvista as pv


def load_cross_sections(xs_file: str, z: float = 0.0) -> pv.PolyData:
    """Load a cross-section file into a PyVista line mesh.

    two whitespace separated values (easting, northing). 
    Consecutive row pairs define one cross-section line segment.
    """
    coords = np.loadtxt(xs_file).reshape(-1, 2, 2)
    points, lines = [], []
    offset = 0
    for pair in coords:
        pts = np.column_stack([pair, np.full(2, z)])
        points.extend(pts)
        lines += [2, offset, offset + 1]
        offset += 2
    mesh = pv.PolyData()
    mesh.points = np.array(points)
    mesh.lines = np.array(lines)
    return mesh


def add_xs_overlay(pl: pv.Plotter, xs_mesh: pv.PolyData) -> None:
    pl.add_mesh(xs_mesh, color='red', line_width=2, render_lines_as_tubes=True)
