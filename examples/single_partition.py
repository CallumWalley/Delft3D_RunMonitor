from Delft3D_RunMonitor.ugrid_mesh import UGridMesh
import pyvista as pv
import defopt
from netCDF4 import Dataset


def main(*, mapname: str='FlowFM_0000_map.nc', varname: str="mesh2d_waterdepth", time_index: int=0):
    """
    mapname: name of the map NetCDF file
    varname: variable name
    time_index: time index
    """
    ugrid = UGridMesh(mapname)
    bedlevelt0 = ugrid._readField('mesh2d_s1', 0) - ugrid._readField('mesh2d_dg', 0)

    bedlevel = ugrid._readField('mesh2d_s1', time_index) - ugrid._readField('mesh2d_dg', time_index)
    dod = bedlevel - bedlevelt0


    polymesh = ugrid._buildVTKPolyData()
    polymesh.cell_data["waterdepth"] = ugrid._readField('mesh2d_waterdepth', 0)
    polymesh.cell_data["dod"] = dod

    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0, 0)
    pl.add_mesh(polymesh, scalars="mesh2d_waterdepth")
    
    pl.subplot(0, 1)

    pl.add_mesh(polymesh, scalars="dod")


    pl.show()

if __name__ == '__main__':
    defopt.run(main)