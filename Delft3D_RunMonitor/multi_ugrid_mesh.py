import numpy as np
import pyvista as pv
from .ugrid_mesh import UGridMesh
import time


class MultiUGridMesh(UGridMesh):

    def __init__(self, filenames):
        """
        Parameters
        ----------
        filenames : list of map filenames
        """
        self.meshes = [UGridMesh(fn) for fn in filenames]
        self.time = 0
        if len(self.meshes) > 0:
            self.time = self.meshes[0].time

    def to_pyvista(self, varname, time_index):
        """
        Convert mesh to a PyVista PolyData object.
        """
        polydata = pv.merge([m.to_pyvista(varname, time_index) for m in self.meshes])
        return polydata

    def movie(self, varname, moviefile="animation.mp4", clim=None):
        """
        Make movie
        """
        # We could just call movie from UGridMesh but this implementation is 2-3 times faster

        tic = time.time()
        pv.OFF_SCREEN = True

        # get the mesh
        polydata = self.to_pyvista(varname=varname, time_index=0)

        data_ptr = None
        data_ptr = polydata.cell_data.get(varname, None)
        if data_ptr is None:
            data_ptr = polydata.point_data.get(varname, None)
        if data_ptr is None:
            raise RuntimeError(f'ERROR could not find data {varname}')

        data_list = [m._readField(varname=varname, time_index=0) for m in self.meshes]

        plotter = pv.Plotter(off_screen=True)
        plotter.open_movie(moviefile)
        nt = len(self.time)       
        for time_index in range(nt):
            print(f'time index {time_index}')
            plotter.clear()
            # iterate over meshes
            for i in range(len(self.meshes)):
                m = self.meshes[i]
                # read the data
                data_list[i][:] = m._readField(varname=varname, time_index=time_index)
            # merge the data
            data_ptr[:] = np.concatenate(data_list)
            plotter.add_mesh(polydata, scalars=varname, clim=clim)
            plotter.write_frame()
        plotter.close()

        toc = time.time()
        print(f'time to create {nt} frames: {toc - tic:.2f} s ({(toc - tic)/nt:.2f} s/frame)')

    

