import numpy as np
import pyvista as pv
from pathlib import Path

MESH_FORMATS  = {'.stl', '.vtp', '.vtk', '.ply', '.obj'}
IMAGE_FORMATS = {'.png', '.jpg', '.jpeg'}
VIDEO_FORMATS = {'.mp4', '.avi', '.mov'}
GIF_FORMATS = {'.gif'}


def load_cross_sections(xs_file: str, z: float = 0.0) -> tuple:
    """Load a cross-section file into a PyVista line mesh.

    File format: pairs of 'easting northing' rows.
    A # comment line immediately before a pair is used as the section name.
    Consecutive row pairs define one cross-section line segment.

    Returns (mesh, names) where names is a list of section labels (None if
    no name was given for that section).
    """
    names = []
    coord_rows = []
    pending_name = None

    with open(xs_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                pending_name = line.lstrip('#').strip()
            else:
                coord_rows.append([float(x) for x in line.split()])
                if len(coord_rows) % 2 == 0:
                    names.append(pending_name)
                    pending_name = None

    coords = np.array(coord_rows).reshape(-1, 2, 2)
    points, lines, scalars = [], [], []
    offset = 0
    for i, pair in enumerate(coords):
        pts = np.column_stack([pair, np.full(2, z)])
        points.extend(pts)
        lines += [2, offset, offset + 1]
        scalars.extend([i, i])
        offset += 2

    mesh = pv.PolyData()
    mesh.points = np.array(points)
    mesh.lines = np.array(lines)
    mesh.point_data["xs_index"] = np.array(scalars, dtype=float)
    return mesh, names


def add_xs_overlay(pl: pv.Plotter, xs_mesh: pv.PolyData, names: list) -> None:
    n = len(names)
    pl.add_mesh(
        xs_mesh,
        scalars="xs_index",
        cmap="tab10",
        clim=[-0.5, n - 0.5],
        line_width=3,
        render_lines_as_tubes=True,
        show_scalar_bar=False,
    )

    endpoints = xs_mesh.points[1::2]
    labelled = [(pt, name) for pt, name in zip(endpoints, names) if name]
    if labelled:
        mids, labels = zip(*labelled)
        pl.add_point_labels(
            list(mids),
            list(labels),
            font_size=8,
            always_visible=True,
            show_points=False,
            shape=None,
        )
