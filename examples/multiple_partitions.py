from Delft3D_RunMonitor.multi_ugrid_mesh import MultiUGridMesh
import defopt
from glob import glob

def main(*, mappattern: str='FlowFM_*_map.nc', varname: str="mesh2d_waterdepth", time_index: int=0):
    """
    mappattern: glob pattern for the map files
    varname: variable name
    time_index: time index
    """
    mapnames = glob(mappattern)
    mesh = MultiUGridMesh(mapnames)
    mesh.plot(varname, time_index)

if __name__ == '__main__':
    defopt.run(main)