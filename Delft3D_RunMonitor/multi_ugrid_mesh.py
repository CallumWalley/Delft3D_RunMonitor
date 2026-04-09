import numpy as np
import pyvista as pv
from .ugrid_mesh import UGridMesh


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


    

