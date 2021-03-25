#!/usr/bin/env python3

from pystencils.session import *
from lbmpy.session import *

from lbmpy.phasefield_allen_cahn.parameter_calculation import calculate_parameters_rti
from lbmpy.phasefield_allen_cahn.kernel_equations import *
from lbmpy.phasefield_allen_cahn.force_model import MultiphaseForceModel

from pystencils.simp import sympy_cse

import pycuda.autoinit
import pycuda.driver as drv

import pystencils.astnodes as psast
import pystencils as ps
import sympy as sp

# domain
R = 500
domain_size = (254, 198, 198)

# time step
timesteps = 10000

# density of the heavier fluid
rho_H = 0.01
# density of the lighter fluid
rho_L = 1.0
# surface tension
sigma = 0.0001
# mobility
M = 0.02

drho3 = (rho_H - rho_L) / 3
# interface thickness
W = 5
# coeffcient related to surface tension
beta = 12.0 * (sigma / W)
# coeffcient related to surface tension
kappa = 1.5 * sigma * W
# relaxation rate allen cahn (h)
w_c = 1.0 / (0.5 + (3.0 * M))


stencil_phase = get_stencil("D3Q19")
stencil_hydro = get_stencil("D3Q19")  # D3Q15, D3Q19, D3Q27
assert len(stencil_phase[0]) == len(stencil_hydro[0])


# create a datahandling object
dh = ps.create_data_handling(
    (domain_size), periodicity=(True, True, True), parallel=False, default_target="gpu"
)

# fields
g = dh.add_array("g", values_per_cell=len(stencil_hydro))
dh.fill("g", 0.0, ghost_layers=True)
h = dh.add_array("h", values_per_cell=len(stencil_phase))
dh.fill("h", 0.0, ghost_layers=True)

g_tmp = dh.add_array("g_tmp", values_per_cell=len(stencil_hydro))
dh.fill("g_tmp", 0.0, ghost_layers=True)
h_tmp = dh.add_array("h_tmp", values_per_cell=len(stencil_phase))
dh.fill("h_tmp", 0.0, ghost_layers=True)

u = dh.add_array("u", values_per_cell=dh.dim)
dh.fill("u", 0.0, ghost_layers=True)

C = dh.add_array("C")
dh.fill("C", 0.0, ghost_layers=True)


dimensions = len(stencil_phase[0])

# relaxation time and rate
tau = 0.03 + 0.5
s8 = 1 / (tau)

# density for the whole domain
rho = rho_L + (C.center) * (rho_H - rho_L)

# body force
body_force = [0, 1e-6, 0]

method_phase = create_lb_method(
    stencil=stencil_phase, method="srt", relaxation_rate=w_c, compressible=True
)

method_hydro = create_lb_method(
    stencil=stencil_hydro,
    method="mrt",
    weighted=True,
    relaxation_rates=[s8, 1, 1, 1, 1, 1],
    maxwellian_moments=True,
    entropic=True,
)

h_updates = initializer_kernel_phase_field_lb(h, C, u, method_phase, W)
g_updates = initializer_kernel_hydro_lb(g, u, method_hydro)

h_init = ps.create_kernel(
    h_updates, target=dh.default_target, cpu_openmp=True
).compile()
g_init = ps.create_kernel(
    g_updates, target=dh.default_target, cpu_openmp=True
).compile()


# initialize the domain
def Initialize_distributions():
    x0 = (domain_size[0] / 2) + 1
    y0 = (domain_size[1] / 2) + 1

    if dimensions == 2:
        for block in dh.iterate(ghost_layers=True, inner_ghost_layers=False):
            x, y = block.midpoint_arrays
            tmp = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            init_values = 0.5 + 0.5 * np.tanh(
                2.0 * (tmp - R) / W
            )  # + random.uniform(0, 0.01)
            block["C"][:, :] = 0
            block["C"][:, :] = init_values
            # block["g"][100:200, 100:160] = 10.6
            # block["h"][100:200, 100:160] = 10.6
            # block["g_tmp"][100:200, 100:160] = 10.6
            # block["h_tmp"][100:200, 100:160] = 10.6
            block["u"][:, :, 0] = init_values * 0.0001

    if dimensions == 3:
        z0 = (domain_size[2] / 2) + 1

        for block in dh.iterate(ghost_layers=True, inner_ghost_layers=False):
            x, y, z = block.midpoint_arrays
            tmp = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
            init_values = 0.5 + 0.5 * np.tanh(2.0 * (tmp - R) / W)
            block["C"][:, :, :] = init_values

    dh.all_to_gpu()

    dh.run_kernel(h_init)
    dh.run_kernel(g_init)


Initialize_distributions()

force_h = [f / 3 for f in interface_tracking_force(C, stencil_phase, W)]
force_model_h = MultiphaseForceModel(force=force_h)
force_g = hydrodynamic_force(
    g, C, method_hydro, tau, rho_H, rho_L, kappa, beta, body_force
)  # fd_stencil="D2Q9"

method_phase.set_force_model(force_model_h)

allen_cahn_lb = create_lb_update_rule(
    lb_method=method_phase,
    velocity_input=u,
    output={"density": C},
    compressible=True,
    optimization={"symbolic_field": h, "symbolic_temporary_field": h_tmp},
    kernel_type="stream_pull_collide",
    # kernel_type="collide_stream_push",
)

ast_allen_cahn_lb = ps.create_kernel(
    allen_cahn_lb, target=dh.default_target, cpu_openmp=True
)
kernel_allen_cahn_lb = ast_allen_cahn_lb.compile()

hydro_lb_update_rule = get_collision_assignments_hydro(
    lb_method=method_hydro,
    density=rho,
    velocity_input=u,
    force=force_g,
    sub_iterations=2,
    symbolic_fields={"symbolic_field": g,"symbolic_temporary_field": g_tmp},
    kernel_type="collide_stream_push",
)

hydro_lb_update_rule = sympy_cse(hydro_lb_update_rule)

ast_hydro_lb = ps.create_kernel(hydro_lb_update_rule, target="gpu")

kernel_hydro_lb = ast_hydro_lb.compile()


def transformReadOnly(eqs):
    newEqs = []
    counter = 0
    for eq in eqs:
        if isinstance(eq.lhs, ps.field.Field.Access):
            tempSymbol = sp.Symbol("temp" + str(counter))
            counter += 1
            newEqs.append(ps.Assignment(tempSymbol, eq.rhs))
            newEqs.append(
                psast.Conditional(
                    sp.Eq(tempSymbol, 2.2123),
                    psast.Block([ps.Assignment(eq.lhs, tempSymbol)]),
                )
            )
        else:
            newEqs.append(eq)
    return newEqs


def transformWriteOnly(eqs):
    counter = 0.2

    def visit(expr):
        fields = []

        if isinstance(expr, ps.field.Field.Access):
            fields.append(expr)
        for a in expr.args:
            if isinstance(a, ps.field.Field.Access):
                fields.append(a)
            fields.extend(visit(a))
        return fields

    newEqs = []
    for eq in eqs:
        if isinstance(eq, ps.Assignment):
            fields = visit(eq.rhs)
            newEqs.append(eq.subs([(f, 0.2) for f in fields]))
    return newEqs


def get_allen_cahn_ast(block_size, mode=None):
    update_rule = allen_cahn_lb
    if mode == "readOnly":
        update_rule = transformReadOnly(update_rule)

    if mode == "writeOnly":
        update_rule = transformWriteOnly(update_rule)

    return ps.create_kernel(
        update_rule,
        target="gpu",
        gpu_indexing_params={
            "block_size": block_size,
            "permute_block_size_dependent_on_layout": False,
        },
    )


def get_hydro_lb_ast(block_size, mode=None):
    update_rule = hydro_lb_update_rule
    if mode == "readOnly":
        update_rule = transformReadOnly(update_rule)
    if mode == "writeOnly":
        update_rule = transformWriteOnly(update_rule)
    return ps.create_kernel(
        update_rule,
        target="gpu",
        gpu_indexing_params={
            "block_size": block_size,
            "permute_block_size_dependent_on_layout": False,
        },
    )


def get_lbm_ast(block_size, mode=None, kernel_type="stream_pull_collide"):
    options = {
        "method": "srt",
        "stencil": "D2Q9",
        "relaxation_rate": w_c,
        "kernel_type": kernel_type,
    }
    lb_method = create_lb_method(**options)

    update_rule = create_lb_update_rule(
        lb_method=lb_method,
        optimization={"symbolic_field": h, "symbolic_temporary_field": h_tmp},
        **options
    )

    if mode == "readOnly":
        update_rule = transformReadOnly(update_rule)
    elif mode == "writeOnly":
        update_rule = transformWriteOnly(update_rule)

    return ps.create_kernel(
        update_rule,
        target="gpu",
        gpu_indexing_params={
            "block_size": block_size,
            "permute_block_size_dependent_on_layout": False,
        },
    )
