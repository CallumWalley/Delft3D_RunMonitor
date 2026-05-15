import numpy as np
import pyvista as pv
from pathlib import Path

MESH_FORMATS  = {'.stl', '.vtp', '.vtk', '.ply', '.obj'}
IMAGE_FORMATS = {'.png', '.jpg', '.jpeg'}
VIDEO_FORMATS = {'.mp4', '.avi', '.mov'}
GIF_FORMATS = {'.gif'}


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


def export_frames(output: str, frames: list, update_frame, polymesh: pv.PolyData,
                  plotter: pv.Plotter) -> None:
    """Export rendered frames to file. Format is determined by extension.

    :param output: Output file path. Extension sets format.
    :param frames: List of time indices to export.
    :param update_frame: Callable that updates the scene for a given time index.
    :param polymesh: PyVista mesh (used for 3D mesh export formats).
    :param plotter: PyVista plotter (used for video and image formats).

    Supported formats:
        Video: .mp4 .avi .mov  
        Gif: .gif            
        Image: .png .jpg .jpeg
        3D: .stl .vtp .vtk .ply .obj
    """
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()

    def numbered(ti):
        return out.with_stem(f"{out.stem}_{ti:04d}") if len(frames) > 1 else out

    def progress(i):
        print(f"  frame {i + 1}/{len(frames)}", end="\r", flush=True)

    if suffix in MESH_FORMATS:
        for i, ti in enumerate(frames):
            progress(i)
            update_frame(ti)
            polymesh.save(str(numbered(ti)))
    elif suffix in IMAGE_FORMATS:
        for i, ti in enumerate(frames):
            progress(i)
            update_frame(ti)
            plotter.screenshot(str(numbered(ti)))
    else:
        if suffix in GIF_FORMATS:
            opener = plotter.open_gif
        elif suffix in VIDEO_FORMATS:
            opener = plotter.open_movie
        else:
            raise ValueError(f"Unknown output format {suffix}")

        opener(output)
        for i, ti in enumerate(frames):
            progress(i)
            update_frame(ti)
            plotter.write_frame()
        plotter.close()

    print(f"\nSaved {len(frames)} frame(s) → {output}")
