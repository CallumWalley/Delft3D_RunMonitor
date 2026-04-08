import numpy as np
from netCDF4 import Dataset
import pyvista as pv


class UGridMesh:

    def __init__(self, filename):

        # Data containers
        self.x = None
        self.y = None
        self.z = None
        self.face_nodes = None
        self.edge_nodes = None
        self.variables = {}
        self.nc = Dataset(filename, "r")
        self._readMesh()

    def _readMesh(self):
        """
        Read the UGrid mesh (points and connectivity)
        """
        # --- Node coordinates ---
        self.x = self.nc.variables["mesh2d_node_x"][:]
        self.y = self.nc.variables["mesh2d_node_y"][:]

        if "mesh2d_node_z" in self.nc.variables:
            self.z = self.nc.variables["mesh2d_node_z"][:]
        else:
            self.z = None

        # --- Edge connectivity ---
        edge_var = self.nc.variables["mesh2d_edge_nodes"]
        self.edge_nodes = edge_var[:].astype(np.int64)

        # Convert to 0-based indexing if needed
        start_index = getattr(edge_var, "start_index", 0)
        if start_index != 0:
            self.edge_nodes -= start_index

        # --- Face connectivity ---
        face_var = self.nc.variables["mesh2d_face_nodes"]
        self.face_nodes = face_var[:].astype(np.int64)

        # Handle fill values (ragged faces)
        fill_value = getattr(face_var, "_FillValue", None)
        if fill_value is not None:
            self.face_nodes = np.where(
                self.face_nodes == fill_value, -1, self.face_nodes
            )

        # Convert to 0-based indexing
        start_index = getattr(face_var, "start_index", 0)
        if start_index != 0:
            self.face_nodes = np.where(
                self.face_nodes >= 0,
                self.face_nodes - start_index,
                self.face_nodes
            )

    def _readField(self, varname, time_index):
        """
        Read the field values at time time_index from the NetCDF file
        """
        return self.nc.variables[varname][time_index, :]
    
    def _buildVTKPolyData(self):
        """
        Build the VTK PolyData object
        """
        # Points (N, 3)
        points = np.column_stack((self.x, self.y, self.z))

        # Build faces in VTK format:
        # [npts, p0, p1, ..., npts, ...]
        faces_list = []
        for face in self.face_nodes:
            valid = face[face >= 0]  # remove fill values
            if len(valid) < 3:
                continue
            faces_list.append(np.concatenate(([len(valid)], valid)))

        if len(faces_list) == 0:
            raise ValueError("No valid faces found")

        faces = np.hstack(faces_list)

        return pv.PolyData(points, faces)

    
    def to_pyvista(self, varname, time_index):
        """
        Convert mesh to a PyVista PolyData object.
        """
        polydata = self._buildVTKPolyData()

        # Read and add the fields
        v = self.nc.variables[varname]
        data = self._readField(varname, time_index)
        location = getattr(v, 'location', 'node')
        if location == 'node':
            polydata.point_data[varname] = data
        elif location == 'face':
            polydata.cell_data[varname] = data
        else:
            raise ValueError(f"ERROR: location {v['location']} is not supported")

        return polydata
    
    def plot(self, varname, time_index, cmap='plasma', clim=None, show_edges=False):
        """
        Plot field at time index
        """
        polydata = self.to_pyvista(varname, time_index)
        plotter = pv.Plotter()
        plotter.add_mesh(polydata, show_edges=show_edges, clim=clim, cmap=cmap)
        plotter.show()
