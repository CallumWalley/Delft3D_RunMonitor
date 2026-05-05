from Delft3D_RunMonitor import MultiUGridMesh, load_cross_sections, add_xs_overlay
from glob import glob
import defopt
import time
import numpy as np
import pyvista as pv


def main(*, mappattern: str='FlowFM_*_map.nc', start_time: int=0, end_time: int=None,
         cmin: float=None, cmax: float=None, xs_file: str=None):
    """
    mappattern: glob pattern for map files (or a single filename)
    start_time: first frame to plot
    end_time: last frame to plot (exclusive, default: all)
    cmin: minimum colour scale value
    cmax: maximum colour scale value
    xs_file: path to cross-section point-pair file (optional)
    """
    ugrid = MultiUGridMesh(sorted(glob(mappattern)))
    end_time = end_time or len(ugrid.time)
    clim_wd  = [cmin, cmax] if cmin is not None and cmax is not None else [0, 2]
    clim_dod = [cmin, cmax] if cmin is not None and cmax is not None else [-2, 2]

    bedlevel_t0 = ugrid.readField('mesh2d_s1', 0) - ugrid.readField('mesh2d_dg', 0)
    polymesh = ugrid.to_pyvista()
    polymesh.cell_data["waterdepth"] = ugrid.readField('mesh2d_waterdepth', 0)
    polymesh.cell_data["dod"] = np.zeros_like(bedlevel_t0)

    bar = {'vertical': True, 'position_x': 0.9, 'position_y': 0.05, 'width': 0.05, 'height': 0.9}
    xs_mesh = load_cross_sections(xs_file) if xs_file else None

    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0, 0)
    pl.add_mesh(polymesh, scalars="waterdepth", clim=clim_wd, scalar_bar_args=bar)
    if xs_mesh:
        add_xs_overlay(pl, xs_mesh)
    pl.subplot(0, 1)
    pl.add_mesh(polymesh, scalars="dod", clim=clim_dod, cmap="bwr", scalar_bar_args=bar)
    if xs_mesh:
        add_xs_overlay(pl, xs_mesh)
    pl.show(auto_close=False)

    for time_index in range(start_time, end_time):
        bedlevel_t = ugrid.readField('mesh2d_s1', time_index) - ugrid.readField('mesh2d_dg', time_index)
        polymesh.cell_data["waterdepth"] = ugrid.readField('mesh2d_waterdepth', time_index)
        polymesh.cell_data["dod"] = bedlevel_t - bedlevel_t0
        pl.subplot(0, 0)
        pl.add_title(f"t={time_index}/{end_time}", color='black')
        pl.render()
        time.sleep(0.1)

    pl.close()


if __name__ == '__main__':
    defopt.run(main)
