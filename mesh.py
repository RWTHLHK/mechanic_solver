from dolfinx import mesh, plot, fem
from mpi4py import MPI
import numpy as np
from mpi4py.MPI import Comm
from numpy.typing import ArrayLike, DTypeLike
from dolfinx.mesh import Mesh, GhostMode, CellType
from typing import Any, Optional, Callable, Dict
import pyvista

class MeshGenerator:
    def __init__(self, dimension:int, shape_definition=None, mesh_func: Optional[Callable]=None, params: Dict = None):
        """
        General mesh generator initialization.

        Parameters:
        - dimension (int): The dimension of the mesh, supporting 2D and 3D.
        - shape_definition (dict or None): User-defined shape description, such as boundary points, etc.
        - mesh_func (callable or None): Custom mesh generation function, which takes dimension and shape_definition to generate a mesh.
        - kwargs: Other parameters used for mesh generation.
        """
        self.dimension = dimension
        self.shape_definition = shape_definition
        self.mesh_func = mesh_func
        self.params = params
        self.mesh = None

    def generate_mesh(self) -> mesh.Mesh:
        """
        Generate mesh based on user-defined or default rules.
        """
        if self.mesh_func:
            # Use the user-provided mesh generation function
            mesh = self.mesh_func(**self.params)
            if self.dimension != mesh.geometry.dim:
                raise ValueError("Dimension of mesh doesn't fit generator dimension!")
            
            self.mesh = mesh
            return self.mesh
        else:
            # Default mesh generation logic
            if self.dimension == 2:
                return self._generate_default_2d_mesh()
            elif self.dimension == 3:
                return self._generate_default_3d_mesh()
            else:
                raise ValueError("Dimension must be 2 or 3.")

    def _generate_default_2d_mesh(self):
        """
        Default 2D mesh generation logic, using boundary points in the shape definition.
        """
        if not self.shape_definition or 'vertices' not in self.shape_definition:
            raise ValueError("2D mesh requires 'vertices' in shape_definition.")
        
        vertices = self.shape_definition['vertices']
        domain = mesh.create_polygon(MPI.COMM_WORLD, vertices, cell_type=mesh.CellType.triangle)
        self.mesh = domain
        return self.mesh

    def _generate_default_3d_mesh(self):
        """
        Default 3D mesh generation logic, generating a cylinder or other default shape based on the definition.
        """
        if self.shape_definition and 'radius' in self.shape_definition and 'height' in self.shape_definition:
            radius = self.shape_definition['radius']
            height = self.shape_definition['height']
            self.mesh = mesh.create_cylinder(MPI.COMM_WORLD, [0, 0, 0], [0, 0, height], radius, 32)
            return self.mesh
        else:
            raise ValueError("3D mesh requires 'radius' and 'height' in shape_definition for default cylinder.")
        
    def show(self):
        p = pyvista.Plotter()
        V = fem.functionspace(self.mesh, ("Lagrange", 1, (self.dimension, )))
        topology, cell_types, geometry = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        # Attach vector values to grid and warp grid by vector
        p.add_mesh(grid, style="wireframe", color="k")
        p.show_axes()
        p.show()

if __name__ == "__main__":

    def gen_box(
        comm: Comm,
        points: list[ArrayLike],
        n: list[int],
        cell_type: Any = CellType.tetrahedron,
        dtype: DTypeLike = mesh.default_real_type,
        ghost_mode: Any = GhostMode.shared_facet,
        partitioner: Optional[Any] = None
    ) -> Mesh:
        """
        Create a box mesh with the specified parameters.

        Parameters:
        - comm (Comm): The MPI communicator for the mesh.
        - points (list[ArrayLike]): A list of corner points defining the box in the format [[x0, y0, z0], [x1, y1, z1]].
        - n (list[int]): Number of cells in each direction, e.g., [nx, ny, nz].
        - cell_type (Any): The cell type for the mesh (default is CellType.tetrahedron).
        - dtype (DTypeLike): Data type for the mesh coordinates (default is `default_real_type` from `dolfinx`).
        - ghost_mode (Any): Ghost cell mode (default is GhostMode.shared_facet).
        - partitioner (Optional[Any]): Optional partitioner to be used for mesh partitioning.

        Returns:
        - Mesh: The created box mesh.
        """
        return mesh.create_box(comm, points, n, cell_type=cell_type, dtype=dtype, ghost_mode=ghost_mode, partitioner=partitioner)
    
    params = {
        "comm": MPI.COMM_WORLD,
        "points": [np.array([0, 0, 0]), np.array([1, 1, 1])],
        "n": [8, 8, 8],
        "cell_type": mesh.CellType.hexahedron
    }

    mesh_generator = MeshGenerator(dimension=3,mesh_func=gen_box, params=params)
    box = mesh_generator.generate_mesh()
    mesh_generator.show()



        

