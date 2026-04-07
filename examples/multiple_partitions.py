from Delft3D_RunMonitor.multi_ugrid_mesh import MultiUGridMesh
import defopt
from glob import glob

def main(*, mappattern: str='FlowFM_*_map.nc', varnames: list[str]=["mesh2d_waterdepth"], time_index: int=0):
    """
    mappattern: glob pattern for the map files
    varnames: list of variable names
    time_index: time index
    """
    mapnames = glob(mappattern)
    mesh = MultiUGridMesh(mapnames, varnames=varnames, time_index=time_index)
    mesh.plot()

if __name__ == '__main__':
    defopt.run(main)