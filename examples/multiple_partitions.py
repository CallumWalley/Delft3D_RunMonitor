from Delft3D_RunMonitor import *
import sys

mesh = MultiUGridMesh(sorted(sys.argv[1:]))
Viewer([PlotView(mesh, varname='mesh2d_waterdepth', clim=[0, 5])]).run()
