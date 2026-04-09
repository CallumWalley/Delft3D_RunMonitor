from Delft3D_RunMonitor.multi_ugrid_mesh import MultiUGridMesh
import defopt
from glob import glob

def main(*, mappattern: str='FlowFM_*_map.nc', varname: str="mesh2d_waterdepth", cmin: float=None, cmax: float=None):
    """
    mappattern: glob pattern for the map files
    varname: variable name
    cmin: min float colourmap value
    cmax: max float colourmap value
    """
    mapnames = glob(mappattern)
    mesh = MultiUGridMesh(mapnames)
    clim = None
    if type(cmin) is float and type(cmax) is float:
        clim = [float(cmin), float(cmax)]
    mesh.movie(varname, clim=clim)

if __name__ == '__main__':
    defopt.run(main)