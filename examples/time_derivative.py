from Delft3D_RunMonitor import *
import sys
import numpy as np

mesh = MultiUGridMesh(sorted(sys.argv[1:]))
time = mesh.meshes[0].time[:]

def dh_dt(mesh, ti):
    if ti == 0:
        return np.zeros_like(mesh.readField('mesh2d_waterdepth', 0))
    f0 = mesh.readField('mesh2d_waterdepth', ti - 1)
    f1 = mesh.readField('mesh2d_waterdepth', ti)
    return (f1 - f0) / (time[ti] - time[ti - 1])

Viewer([PlotView(mesh, field_fn=dh_dt, clim=[-0.1, 0.1], cmap='bwr',
                 title='dh/dt (m/s)')]).run()
