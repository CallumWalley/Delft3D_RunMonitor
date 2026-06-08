from glob import glob
import sys

from Delft3D_RunMonitor import MultiUGridMesh, PlotView, Viewer, CrossSectionOverlay

def _dod(mesh, ti):
    """Bed level change relative to t=0 (depth of difference)."""
    bed_t0 = mesh.readField("mesh2d_s1", 0) - mesh.readField("mesh2d_dg", 0)
    bed_ti = mesh.readField("mesh2d_s1", ti) - mesh.readField("mesh2d_dg", ti)
    return bed_ti - bed_t0


mesh = MultiUGridMesh(sorted(glob(sys.argv[2])))
overlays = [CrossSectionOverlay(sorted(glob(sys.argv[1]))[0])]

Viewer([
    PlotView(mesh, "mesh2d_waterdepth", title="Water Depth (m)",
             overlays=overlays, clim=[0, 1]),
    PlotView(mesh, field_fn=_dod, title="Depth of Difference (m)",
                cmap="bwr", clim=[-2, 2], overlays=overlays),
]).run()