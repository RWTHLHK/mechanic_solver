import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
from bcs import BoundaryConditionGenerator
l = 1
W = 0.2
mu = 1
rho = 1
delta = W / l
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma

domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([l, W, W])],
                         [20, 6, 6], cell_type=mesh.CellType.hexahedron)
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

def left_boundary(x):
    return np.isclose(x[0],0)

bc_generator = BoundaryConditionGenerator(domain, V)
bc_generator.add_boundary_condition(
    bc_type="Dirichlet", 
    value= np.array([0,0,0], dtype=default_scalar_type),
    boundary_marker=left_boundary
)
bcs = bc_generator.get_boundary_conditions()
bcs.append(None)
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
ds = ufl.Measure("ds", domain=domain)

def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(T, v) * ds

t_end = 1.0  # 总加载时间
num_steps = 50  # 时间步数
dt = t_end / num_steps  # 每个时间步的时长
load_magnitude = 0.1  # 总位移量
fdim = domain.topology.dim - 1
for step in range(num_steps):
    current_time = step * dt

    # 计算当前位移加载量
    displacement_value = load_magnitude * (current_time / t_end)  # 线性增加
    current_top_position = l + displacement_value
    def dynamic_top_boundary(x):
        return np.isclose(x[0], current_top_position)
    
    # print("current position is: ", current_top_position)
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, dynamic_top_boundary)
    u_t = np.array([current_top_position,0,0], dtype=default_scalar_type)
    bc_top = fem.dirichletbc(u_t,fem.locate_dofs_topological(V,fdim,boundary_facets),V)
    bcs[1] = bc_top
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    # Create plotter and pyvista grid
    p = pyvista.Plotter(off_screen=True)
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    # # Attach vector values to grid and warp grid by vector
    grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
    print(f"step: {step}")
    print(grid["u"])
    # actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    # warped = grid.warp_by_vector("u", factor=1.5)
    # actor_1 = p.add_mesh(warped, show_edges=True)
    # p.show_axes()
    # p.screenshot(f"mesh_plot_step{step}.png")
    # p.close()

# # Attach vector values to grid and warp grid by vector
# grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
# actor_0 = p.add_mesh(grid, style="wireframe", color="k")
# warped = grid.warp_by_vector("u", factor=1.5)
# actor_1 = p.add_mesh(warped, show_edges=True)
# p.show_axes()
# p.show()

# with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     uh.name = "Deformation"
#     xdmf.write_function(uh)
