from Delft3D_RunMonitor import MultiUGridMesh, load_cross_sections, add_xs_overlay
import defopt
import time
import numpy as np
import pyvista as pv

from glob import glob


def main(*, mappattern: str='FlowFM_*_map.nc', start_time: int=0, end_time: int=None, cmin: float=None, cmax: float=None, xs_file: str=None):
    """
    mappattern: glob pattern for the map files
    start_time: first frame to plot
    end_time: last frame to plot (exclusive, default: all)
    xs_file: path to cross-section point-pair file (optional)
    """
    mapnames = glob(mappattern)
    ugrid = MultiUGridMesh(mapnames)
    clim = None
    if type(cmin) is float and type(cmax) is float:
        clim = [float(cmin), float(cmax)]

    end_time = len(ugrid.meshes[0].time) if end_time is None else end_time


    bedlevel_t0 = np.concatenate([m.readField('mesh2d_s1', 0) - m.readField('mesh2d_dg', 0) for m in ugrid.meshes])

    polymesh = pv.merge([m._buildVTKPolyData() for m in ugrid.meshes])
    polymesh.cell_data["waterdepth"] = np.concatenate([ m.readField('mesh2d_waterdepth', 0) for m in ugrid.meshes ])
    polymesh.cell_data["dod"] = np.zeros_like(bedlevel_t0)

    scalar_bar_args = {'vertical': True, 'position_x': 0.9, 'position_y': 0.05, 'width': 0.05, 'height': 0.9}

    xs_mesh = load_cross_sections(xs_file) if xs_file else None

    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0, 0)
    pl.add_mesh(polymesh, scalars="waterdepth", clim=clim or [0, 2], scalar_bar_args=scalar_bar_args)
    if xs_mesh:
        add_xs_overlay(pl, xs_mesh)

    pl.subplot(0, 1)
    pl.add_mesh(polymesh, scalars="dod", clim=clim or [-2, 2], cmap="bwr", scalar_bar_args=scalar_bar_args)
    if xs_mesh:
        add_xs_overlay(pl, xs_mesh)

    pl.show(auto_close=False)

    for time_index in range(start_time, end_time):
        bedlevel_t = np.concatenate([m.readField('mesh2d_s1', time_index) - m.readField('mesh2d_dg', time_index) for m in ugrid.meshes])
        dod = np.subtract(bedlevel_t, bedlevel_t0)
        polymesh.cell_data["waterdepth"] = np.concatenate([ m.readField('mesh2d_waterdepth', time_index) for m in ugrid.meshes ])
        polymesh.cell_data["dod"] = dod

        pl.subplot(0, 0)
        pl.add_title(f"t={time_index}/{end_time}", color='black')
        pl.render()
        time.sleep(0.1)

    pl.close()

if __name__ == '__main__':
    import sys
    import defopt
    defopt.run(main)