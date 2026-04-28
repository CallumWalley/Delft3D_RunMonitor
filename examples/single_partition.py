from Delft3D_RunMonitor.ugrid_mesh import UGridMesh
import pyvista as pv
import defopt
from netCDF4 import Dataset
import time
import numpy as np


def main(*, mapname: str='FlowFM_0000_map.nc', start_time: int=0, end_time: int=None):
    """
    mapname: name of the map NetCDF file
    varname: variable name
    time_index: time index
    """
    ugrid = UGridMesh(mapname)
    bedlevelt0 = ugrid._readField('mesh2d_s1', 0) - ugrid._readField('mesh2d_dg', 0)

    pl = pv.Plotter(shape=(1, 2))
    polymesh = ugrid._buildVTKPolyData()

    # Add initial meshes to each subplot
    pl.subplot(0, 0)
    waterdepth = pl.add_mesh(polymesh, scalars=ugrid._readField('mesh2d_waterdepth', 0), name="waterdepth")
    pl.subplot(0, 1)
    dod = pl.add_mesh(polymesh, scalars=np.zeros_like(bedlevelt0), name="dod")

    pl.show(auto_close=False)  # Keep window open

    # Determine the number of timesteps
    end_time = len(ugrid.time) if end_time is None else end_time
    
    time_indices = range(start_time, end_time)

    # Set initial title for the first subplot only
    pl.subplot(0, 0)
    pl.add_title(f"T: {start_time}", color='black')

    for time_index in time_indices:
        bedlevel = ugrid._readField('mesh2d_s1', time_index) - ugrid._readField('mesh2d_dg', time_index)
        dod = bedlevel - bedlevelt0

        # Update cell data on the mesh directly
        polymesh.cell_data["waterdepth"] = ugrid._readField('mesh2d_waterdepth', time_index)
        polymesh.cell_data["dod"] = dod

        # Update title for the first subplot only
        pl.subplot(0, 0)
        pl.add_title(f"t={time_index}", color='black')

        pl.render()
        time.sleep(0.1)

    pl.close()

if __name__ == '__main__':
    defopt.run(main)