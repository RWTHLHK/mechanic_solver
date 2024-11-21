import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
from typing import Callable, Dict, Any, Union, List

class BoundaryConditionGenerator:
    def __init__(self, domain_mesh: mesh.Mesh, function_space: fem.FunctionSpace):
        """
        Initializes the General Boundary Condition Generator.

        Parameters:
        - domain_mesh (mesh.Mesh): The mesh on which the boundary conditions are defined.
        - function_space (fem.FunctionSpace): The function space where the boundary conditions are applied.
        """
        self.domain_mesh = domain_mesh
        self.function_space = function_space
        self.boundary_conditions = []

    def add_boundary_condition(self, bc_type: str, value: Union[float, Callable], 
                               boundary_marker: Callable, method: str = "topological", **kwargs):
        """
        Adds a boundary condition to the generator.

        Parameters:
        - bc_type (str): The type of boundary condition, e.g., "Dirichlet" or "Neumann".
        - value (float or Callable): The value or function defining the boundary condition.
          Can be a constant or a callable function returning values on the boundary.
        - boundary_marker (Callable): A function that marks the boundary where this condition applies.
          It should return True for points on the boundary.
        - method (str): Method to apply the boundary condition ("topological" or "geometrical").
          - "topological": For identifying boundaries based on mesh topology.
          - "geometrical": For identifying boundaries based on geometry.
        - kwargs: Additional arguments for condition customization.
        """
        # Validate boundary condition type
        if bc_type not in ["Dirichlet", "Neumann"]:
            raise ValueError("Unsupported boundary condition type. Use 'Dirichlet' or 'Neumann'.")

        # Create the boundary condition based on type
        if bc_type == "Dirichlet":
            # Dirichlet condition is directly applied on function space boundary
            bc = self._create_dirichlet_bc(value, boundary_marker, method, **kwargs)
        elif bc_type == "Neumann":
            # Neumann condition will be applied in the weak form, store as data
            bc = {"type": "Neumann", "value": value, "boundary_marker": boundary_marker, "method": method}

        self.boundary_conditions.append(bc)

    def _create_dirichlet_bc(self, value, boundary_marker, method, **kwargs):
        """
        Creates a Dirichlet boundary condition.
        """
        # Convert value to a constant or Function depending on its type
        if isinstance(value, (float, int, np.ndarray)):
            value_expr = value
        elif callable(value):
            value_expr = fem.Function(self.function_space)
            value_expr.interpolate(value)
        else:
            raise ValueError("Unsupported value type for Dirichlet boundary condition.")

        # Create the Dirichlet boundary condition
        fdim = self.domain_mesh.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(self.domain_mesh, fdim, boundary_marker)
        return fem.dirichletbc(value_expr, fem.locate_dofs_topological(self.function_space, fdim, boundary_facets), self.function_space)

    def get_boundary_conditions(self) -> List[Any]:
        """
        Retrieves the list of boundary conditions for the problem.

        Returns:
        - List of boundary conditions ready to be applied to a problem.
        """
        return [bc for bc in self.boundary_conditions if isinstance(bc,fem.bcs.DirichletBC)]

    def apply_neumann_conditions(self, v):
        """
        Applies Neumann boundary conditions in the weak form by returning the Neumann contributions to the form.

        Parameters:
        - v (ufl.TestFunction): The test function of the weak form.

        Returns:
        - Neumann contributions to be added to the weak form.
        """
        neumann_terms = []
        for bc in self.boundary_conditions:
            if not isinstance(bc, fem.bcs.DirichletBC):
                ds = ufl.Measure("ds", domain=self.domain_mesh, subdomain_data=bc["boundary_marker"])
                neumann_terms.append(bc["value"] * v * ds)
        return sum(neumann_terms)

if __name__ == "__main__":
    # Example Usage
    domain_mesh = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = fem.functionspace(domain_mesh, ("Lagrange", 1,(domain_mesh.geometry.dim, )))

    # Initialize the boundary condition generator
    bc_generator = BoundaryConditionGenerator(domain_mesh, V)

    # Define boundary marker functions
    def left_boundary(x):
        return np.isclose(x[0], 0)

    def top_boundary(x):
        return np.isclose(x[1], 1)

    # Add Dirichlet condition on the left boundary
    bc_generator.add_boundary_condition(
        bc_type="Dirichlet", 
        value= np.array([0,0], dtype=default_scalar_type),
        boundary_marker=left_boundary
    )

    # Add Neumann condition on the top boundary
    bc_generator.add_boundary_condition(
        bc_type="Neumann", 
        value=np.array([0,0], dtype=default_scalar_type),
        boundary_marker=top_boundary
    )

    # Retrieve Dirichlet conditions for applying to problem
    dirichlet_bcs = bc_generator.get_boundary_conditions()

    # Example of applying Neumann conditions in the weak form
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    f = ufl.Constant(domain_mesh, 0)  # Source term for illustration
    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # Add Neumann contributions to weak form
    F += bc_generator.apply_neumann_conditions(v)

