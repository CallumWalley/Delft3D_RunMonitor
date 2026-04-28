from Delft3D_RunMonitor.multi_ugrid_mesh import MultiUGridMesh
import defopt
import pyvista as pv
import numpy as np


from glob import glob

def main(*, mappattern: str='FlowFM_*_map.nc', time_index: int=0, cmin: float=None, cmax: float=None):
    """
    mappattern: glob pattern for the map files
    varname: variable name
    time_index: time index
    """
    mapnames = glob(mappattern)
    ugrid = MultiUGridMesh(mapnames)
    clim = None
    if type(cmin) is float and type(cmax) is float:
        clim = [float(cmin), float(cmax)]

    bedlevel_t0 = np.concatenate([m._readField('mesh2d_s1', 0) - m._readField('mesh2d_dg', 0) for m in ugrid.meshes])
    bedlevel_t = np.concatenate([m._readField('mesh2d_s1', time_index) - m._readField('mesh2d_dg', time_index) for m in ugrid.meshes])
    dod = np.subtract(bedlevel_t, bedlevel_t0)


    polymesh = pv.merge([m._buildVTKPolyData() for m in ugrid.meshes])
    polymesh.cell_data["waterdepth"] = np.concatenate([ m._readField('mesh2d_waterdepth', 0) for m in ugrid.meshes ])
    polymesh.cell_data["dod"] = dod

    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0, 0)
    pl.add_mesh(polymesh, scalars="waterdepth", clim=clim)
    
    pl.subplot(0, 1)
    pl.add_mesh(polymesh, scalars="dod", clim=clim)


    pl.show()

if __name__ == '__main__':
    defopt.run(main)