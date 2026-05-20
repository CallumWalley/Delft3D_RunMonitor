from Delft3D_RunMonitor import MultiUGridMesh
from typing import List
import defopt

def main(*, mapnames: List[str]=['FlowFM_0001_map.nc', 'FlowFM_0002_map.nc'], varname: str="mesh2d_waterdepth", 
            cmin: float=None, cmax: float=None,
            t0: int=0, t1: int=-1):
    """
    mapnames: list of map file names
    varname: variable name
    cmin: min float colourmap value
    cmax: max float colourmap value
    t0: min time index
    t1: one beyond last time index
    """
    mesh = MultiUGridMesh(mapnames)
    clim = None
    if type(cmin) is float and type(cmax) is float:
        clim = [float(cmin), float(cmax)]
    mesh.movie(varname, clim=clim, t0=t0, t1=t1)


def cli():
    defopt.run(main)


if __name__ == '__main__':
    cli()
