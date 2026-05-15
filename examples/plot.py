from Delft3D_RunMonitor import MultiUGridMesh, load_cross_sections, add_xs_overlay, export_frames
from glob import glob
import defopt
import time
import numpy as np
import pyvista as pv


def main(*, mappattern: str='FlowFM_*_map.nc', start_time: int=0,
         end_time: int=None, step: int=1, cmin: float=None, cmax: float=None,
         xs_file: str=None, output: str=None):
    """
    Plot one or more map files.

    Args:
        mappattern: Glob pattern for map files, or a single filename.
        start_time: First frame to plot.
        end_time: Last frame to plot (exclusive). Defaults to all frames.
        step: Number of time steps to advance per keypress or animation frame.
        cmin: Minimum colour scale value.
        cmax: Maximum colour scale value.
        xs_file: Optional path to a cross-section point-pair file.
        output: Save to file instead of opening interactive window.
                Extension sets format: .mp4 for video, .gif for GIF.
    """

    ugrid = MultiUGridMesh(sorted(glob(mappattern)))

    end_time = end_time or len(ugrid.time)

    clim_wd = [cmin, cmax] \
        if cmin is not None and cmax is not None else [0, 2]

    clim_dod = [cmin, cmax] \
        if cmin is not None and cmax is not None else [-2, 2]

    bedlevel_t0 = (
        ugrid.readField('mesh2d_s1', 0)
        - ugrid.readField('mesh2d_dg', 0)
    )

    polymesh = ugrid.to_pyvista()

    polymesh.cell_data["waterdepth"] = \
        ugrid.readField('mesh2d_waterdepth', 0)

    polymesh.cell_data["dod"] = np.zeros_like(bedlevel_t0)

    bar = {
        'vertical': True,
        'position_x': 0.9,
        'position_y': 0.05,
        'width': 0.05,
        'height': 0.9
    }

    xs_mesh = load_cross_sections(xs_file) if xs_file else None

    pl = pv.Plotter(shape=(1, 2), off_screen=bool(output))

    pl.subplot(0, 0)
    pl.add_mesh(
        polymesh,
        scalars="waterdepth",
        clim=clim_wd,
        scalar_bar_args=bar
    )

    if xs_mesh:
        add_xs_overlay(pl, xs_mesh)

    pl.subplot(0, 1)
    pl.add_mesh(
        polymesh,
        scalars="dod",
        clim=clim_dod,
        cmap="bwr",
        scalar_bar_args=bar
    )

    if xs_mesh:
        add_xs_overlay(pl, xs_mesh)

    pl.link_views()

    current_time = start_time
    running = False

    text_actor = pl.add_text(
        "",
        position="upper_left",
        font_size=12,
        color="black"
    )

    def update_frame(ti):

        nonlocal current_time

        ti = max(start_time, min(ti, end_time - 1))
        current_time = ti

        bedlevel_t = (
            ugrid.readField('mesh2d_s1', ti)
            - ugrid.readField('mesh2d_dg', ti)
        )

        polymesh.cell_data["waterdepth"] = \
            ugrid.readField('mesh2d_waterdepth', ti)

        polymesh.cell_data["dod"] = \
            bedlevel_t - bedlevel_t0

        text_actor.SetText(
            0,
            f"t = {ti}/{end_time - 1}"
        )

        pl.render()

    def step_forward():
        update_frame(current_time + step)

    def goto_start():
        update_frame(start_time)

    def goto_end():
        update_frame(end_time - 1)

    def stop_running():
        nonlocal running
        running = False

    def run_animation():

        nonlocal running

        if running:
            return

        running = True

        while running and current_time < end_time - 1:

            update_frame(current_time + step)

            pl.update()
            time.sleep(0.1)

        running = False

    if output:
        export_frames(output, list(range(start_time, end_time, step)),
                      update_frame, polymesh, pl)
    else:
        pl.add_key_event("t", step_forward)
        pl.add_key_event("r", run_animation)
        pl.add_key_event("space", stop_running)
        pl.add_key_event("s", goto_start)
        pl.add_key_event("e", goto_end)

        update_frame(start_time)

        print("Keyboard controls:")
        print("  t      step forward")
        print("  r      run animation")
        print("  space  stop animation")
        print("  s      first frame")
        print("  e      last frame")

        pl.show()


if __name__ == '__main__':
    defopt.run(main)
    
