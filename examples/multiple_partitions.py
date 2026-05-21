from Delft3D_RunMonitor import MultiUGridMesh
import defopt
from typing import List

def main(*, mapnames: List[str]=['FlowFM_0001_map.nc', 'FlowFM_0002_map.nc'], varname: str="mesh2d_waterdepth", time_index: int=0, cmin: float=None, cmax: float=None):
    """
    mapnames: list of map files
    varname: variable name
    time_index: time index
    """
    mesh = MultiUGridMesh(mapnames)
    clim = None
    if type(cmin) is float and type(cmax) is float:
        clim = [float(cmin), float(cmax)]
    mesh.plot(varname, time_index, clim=clim)

if __name__ == '__main__':
    defopt.run(main)