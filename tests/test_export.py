import pytest
import numpy as np
import pyvista as pv
from Delft3D_RunMonitor import export_frames

pv.OFF_SCREEN = True


@pytest.fixture
def scene():
    mesh = pv.Plane(i_resolution=4, j_resolution=4).triangulate()
    mesh.cell_data["value"] = np.random.rand(mesh.n_cells)
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh, scalars="value")
    return mesh, pl


def noop(ti):
    pass


@pytest.mark.parametrize("ext", [
    ".png", ".jpg", ".stl", ".vtp", ".vtk", ".ply", ".obj", ".gif", ".mp4"
])
def test_single_frame(tmp_path, scene, ext):
    if ext == ".mp4":
        pytest.importorskip("imageio")
    mesh, pl = scene
    out = tmp_path / f"out{ext}"
    export_frames(str(out), [0], noop, mesh, pl)
    assert out.stat().st_size > 0


@pytest.mark.parametrize("ext", [".png", ".stl", ".vtp"])
def test_multi_frame_numbered(tmp_path, scene, ext):
    mesh, pl = scene
    export_frames(str(tmp_path / f"out{ext}"), [0, 1, 2], noop, mesh, pl)
    for i in [0, 1, 2]:
        assert (tmp_path / f"out_{i:04d}{ext}").stat().st_size > 0
