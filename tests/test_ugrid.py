import pytest
from pathlib import Path
from Delft3D_RunMonitor import UGridMesh, MultiUGridMesh

DATA_DIR = Path(__file__).parent.parent / "data"
SINGLE_FILE = str(DATA_DIR / "FlowFM_0000_map.nc")
ALL_FILES = sorted(str(p) for p in DATA_DIR.glob("FlowFM_*_map.nc"))
FIELD = "mesh2d_waterdepth"


@pytest.fixture(scope="module")
def single_mesh():
    return UGridMesh(SINGLE_FILE)


@pytest.fixture(scope="module")
def multi_mesh():
    return MultiUGridMesh(ALL_FILES)


def test_single_loads(single_mesh):
    assert len(single_mesh.time) > 0


def test_single_readfield_shape(single_mesh):
    data = single_mesh.readField(FIELD, 0)
    assert data.ndim == 1
    assert data.shape[0] > 0


def test_single_to_pyvista(single_mesh):
    poly = single_mesh.to_pyvista()
    assert poly.n_points > 0
    assert poly.n_cells > 0


def test_single_to_pyvista_with_field(single_mesh):
    poly = single_mesh.to_pyvista(FIELD, 0)
    assert FIELD in poly.cell_data or FIELD in poly.point_data


def test_multi_partition_count(multi_mesh):
    assert len(multi_mesh.meshes) == len(ALL_FILES)


def test_multi_loads(multi_mesh):
    assert len(multi_mesh.time) > 0


def test_multi_readfield_larger_than_single(single_mesh, multi_mesh):
    single_data = single_mesh.readField(FIELD, 0)
    multi_data = multi_mesh.readField(FIELD, 0)
    assert multi_data.shape[0] > single_data.shape[0]


def test_multi_to_pyvista(multi_mesh):
    poly = multi_mesh.to_pyvista()
    assert poly.n_points > 0
    assert poly.n_cells > 0
