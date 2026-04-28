from Delft3D_RunMonitor.multi_ugrid_mesh import MultiUGridMesh
import pyvista as pv
import defopt
from netCDF4 import Dataset
import time
import numpy as np

from glob import glob

def main(*, mappattern: str='FlowFM_*_map.nc', start_time: int=0, end_time: int=None, cmin: float=None, cmax: float=None):
    """
    mappattern: glob pattern for the map files
    start_time: first frame to plot
    end_time: last frame to plot (exclusive, default: all)
    """
    mapnames = glob(mappattern)
    ugrid = MultiUGridMesh(mapnames)
    clim = None
    if type(cmin) is float and type(cmax) is float:
        clim = [float(cmin), float(cmax)]

    end_time = len(ugrid.meshes[0].time) if end_time is None else end_time


    bedlevel_t0 = np.concatenate([m._readField('mesh2d_s1', 0) - m._readField('mesh2d_dg', 0) for m in ugrid.meshes])

    polymesh = pv.merge([m._buildVTKPolyData() for m in ugrid.meshes])
    polymesh.cell_data["waterdepth"] = np.concatenate([ m._readField('mesh2d_waterdepth', 0) for m in ugrid.meshes ])
    polymesh.cell_data["dod"] = np.zeros_like(bedlevel_t0)

    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0, 0)
    pl.add_mesh(polymesh, scalars="waterdepth", clim=clim)
    pl.subplot(0, 1)
    pl.add_mesh(polymesh, scalars="dod", clim=clim)

    pl.show(auto_close=False)

    for time_index in range(start_time, end_time):
        bedlevel_t = np.concatenate([m._readField('mesh2d_s1', time_index) - m._readField('mesh2d_dg', time_index) for m in ugrid.meshes])
        dod = np.subtract(bedlevel_t, bedlevel_t0)
        polymesh.cell_data["waterdepth"] = np.concatenate([ m._readField('mesh2d_waterdepth', time_index) for m in ugrid.meshes ])
        polymesh.cell_data["dod"] = dod

        pl.subplot(0, 0)
        pl.add_title(f"t={time_index}/{end_time}", color='black')
        pl.render()
        import time
        time.sleep(0.1)

    pl.close()

if __name__ == '__main__':
    import sys
    import defopt
    defopt.run(main)