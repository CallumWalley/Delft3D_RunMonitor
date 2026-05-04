from Delft3D_RunMonitor import UGridMesh, load_cross_sections, add_xs_overlay
import defopt
import time
import numpy as np
import pyvista as pv


def main(*, mapname: str='FlowFM_0000_map.nc', start_time: int=0, end_time: int=None, xs_file: str=None):
    """
    mapname: name of the map NetCDF file
    start_time: first frame to plot
    end_time: last frame to plot (exclusive, default: all)
    xs_file: path to cross-section point-pair file (optional)
    """
    ugrid = UGridMesh(mapname)
    bedlevelt0 = ugrid.readField('mesh2d_s1', 0) - ugrid.readField('mesh2d_dg', 0)

    mesh_waterdepth = ugrid._buildVTKPolyData().copy()
    mesh_dod = ugrid._buildVTKPolyData().copy()
    mesh_dod.cell_data["dod"] = np.zeros_like(bedlevelt0)

    scalar_bar_args = {'vertical': True, 'position_x': 0.9, 'position_y': 0.05, 'width': 0.05, 'height': 0.9}
    xs_mesh = load_cross_sections(xs_file) if xs_file else None

    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0, 0)
    pl.add_mesh(mesh_waterdepth, scalars=ugrid.readField('mesh2d_waterdepth', 0), clim=[0, 1.5], scalar_bar_args=scalar_bar_args, name="waterdepth")
    if xs_mesh:
        add_xs_overlay(pl, xs_mesh)

    pl.subplot(0, 1)
    pl.add_mesh(mesh_dod, scalars="dod", clim=[-2, 2], cmap="bwr", scalar_bar_args=scalar_bar_args, name="dod")
    if xs_mesh:
        add_xs_overlay(pl, xs_mesh)

    pl.show(auto_close=False)


    # Determine the number of timesteps
    end_time = len(ugrid.time) if end_time is None else end_time
    
    time_indices = range(start_time, end_time)

    for time_index in time_indices:
        bedlevel = ugrid.readField('mesh2d_s1', time_index) - ugrid.readField('mesh2d_dg', time_index)
        dod = bedlevel - bedlevelt0

        # Update cell data on each mesh independently
        mesh_waterdepth.cell_data["waterdepth"] = ugrid.readField('mesh2d_waterdepth', time_index)
        mesh_dod.cell_data["dod"] = dod

        # Update title for the first subplot only
        pl.subplot(0, 0)
        pl.add_title(f"t={time_index}/{end_time}", color='black')

        pl.render()
        
        time.sleep(0.1)

    pl.close()

if __name__ == '__main__':
    defopt.run(main)