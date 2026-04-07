from Delft3D_RunMonitor.ugrid_mesh import UGridMesh
import defopt

def main(*, mapname: str='FlowFM_0000_map.nc', varnames: list[str]=["mesh2d_waterdepth"], time_index: int=0):
    """
    mapname: name of the map NetCDF file
    varnames: list of variable names
    time_index: time index
    """
    mesh = UGridMesh(mapname, varnames=varnames)
    mesh.plot()

if __name__ == '__main__':
    defopt.run(main)