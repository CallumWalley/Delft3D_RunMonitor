"""Interactive water depth / depth-of-difference viewer.

Usage
-----
python plot_dod.py --mappattern "output/FlowFM_*_map.nc"
python plot_dod.py --mappattern "output/FlowFM_*_map.nc" --xs-file data/cross_sections.txt
python plot_dod.py --mappattern "output/FlowFM_*_map.nc" --wd-max 3.0 --dod-max 1.5
python plot_dod.py --mappattern "output/FlowFM_*_map.nc" -b 10 -e 100 --step 2
python plot_dod.py --mappattern "output/FlowFM_*_map.nc" --output animation.mp4
"""

from glob import glob
import defopt

from Delft3D_RunMonitor import MultiUGridMesh, PlotView, Viewer, CrossSectionOverlay


def _dod(mesh, ti):
    """Bed level change relative to t=0 (depth of difference)."""
    bed_t0 = mesh.readField("mesh2d_s1", 0) - mesh.readField("mesh2d_dg", 0)
    bed_ti = mesh.readField("mesh2d_s1", ti) - mesh.readField("mesh2d_dg", ti)
    return bed_ti - bed_t0


def main(*, mappattern: str = "FlowFM_*_map.nc",
         xs_file: str = None,
         wd_max: float = 2.0,
         dod_max: float = 2.0,
         b: int = 0,
         e: int = -1,
         step: int = 1,
         output: str = None):
    """View or export water depth and depth-of-difference panels.

    :param mappattern: Glob pattern for map NetCDF files.
    :param xs_file: Optional cross-section overlay file.
    :param wd_max: Water-depth colour scale maximum.
    :param dod_max: Depth-of-difference colour scale maximum.
    :param b: First time index to display (inclusive).
    :param e: Last time index to display (exclusive; -1 = final step).
    :param step: Frame stride — show/export every N-th time step.
    :param output: If given, export to this file instead of opening interactively.
    """
    mesh = MultiUGridMesh(sorted(glob(mappattern)))
    overlays = [CrossSectionOverlay(xs_file)] if xs_file else []

    viewer = Viewer([
        PlotView(mesh, "mesh2d_waterdepth",
                 clim=[0, wd_max], overlays=overlays),
        PlotView(mesh, field_fn=_dod,
                 clim=[-dod_max, dod_max], cmap="bwr", overlays=overlays),
    ], t0=b, t1=e, step=step)

    if output:
        viewer.export(output)
    else:
        viewer.show()


if __name__ == '__main__':
    defopt.run(main)
