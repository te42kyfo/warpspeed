#!/usr/bin/env python3

import pystencils as ps
import pystencils.astnodes as psast
import sympy as sp
import functools
import numpy as np
from pystencils.simp import sympy_cse_on_assignment_list
import random
from sympy import Symbol, numbered_symbols
from predict_metrics import LaunchConfig


def orderLoads(assignments):
    fa_symbol_iter = numbered_symbols("fa_")
    loads = {}
    new_assignments = []
    for a in assignments:
        for l in a.rhs.atoms(psast.Field.Access):
            if not l in loads:
                loads[l] = next(fa_symbol_iter)

    for l in sorted(loads.keys(), key=lambda c: tuple(reversed(c.offsets))):
        new_assignments.append(ps.Assignment(loads[l], l))

    # random.shuffle(new_assignments)

    for a in assignments:
        new_assignments.append(
            ps.Assignment(a.lhs, a.rhs.subs(zip(loads.keys(), loads.values())))
        )

    # for a in new_assignments:
    #    print(a)
    return new_assignments


def blockThreads(assignments, blocking_factors):
    blockedAssignments = []
    for bx, by, bz in np.ndindex(blocking_factors):

        def shiftIndices(expr):
            if isinstance(expr, psast.Field.Access):
                return expr.get_shifted(bx, by, bz)
            if isinstance(expr, psast.Block):
                return psast.Block([shiftIndices(arg) for arg in expr.args])
            if len(expr.args) > 0:
                return expr.func(*[shiftIndices(arg) for arg in expr.args])
            return expr

        for a in assignments:
            blockedAssignments.append(shiftIndices(a))

    return orderLoads(sympy_cse_on_assignment_list(blockedAssignments))


def getStarAssignments(
    src_field, dst_field, stencil_range, blocking_factors=(1, 1, 1), no_store=False
):
    assignments = []
    v = sp.Symbol("v")
    if stencil_range > -1:
        rhs = src_field[0, 0, 0]
    else:
        rhs = 0.25
    for r in range(1, stencil_range + 1):
        rhs = (
            rhs
            + src_field[0, 0, r]
            + src_field[0, 0, -r]
            + src_field[0, r, 0]
            + src_field[0, -r, 0]
            + src_field[r, 0, 0]
            + src_field[-r, 0, 0]
        )

    if no_store:
        assignments = [
            ps.Assignment(v, rhs * 0.25),
            psast.Conditional(
                sp.Eq(v, 2.2),
                psast.Block([ps.Assignment(dst_field[0, 0, 0], v)]),
            ),
        ]
    else:
        assignments = [ps.Assignment(dst_field[0, 0, 0], rhs * 0.25)]

    if blocking_factors != (1, 1, 1):
        return blockThreads(assignments, blocking_factors)
    else:
        return assignments


def getBoxAssignments(
    src_field, dst_field, stencil_range, blocking_factors=(1, 1, 1), no_store=False
):
    assignments = []
    v = sp.Symbol("v")
    if stencil_range > -1:
        rhs = src_field[0, 0, 0]
    else:
        rhs = 0.25

    for r1 in range(-stencil_range, stencil_range + 1):
        for r2 in range(-stencil_range, stencil_range + 1):
            for r3 in range(-stencil_range, stencil_range + 1):
                if r1 == 0 and r2 == 0 and r3 == 0:
                    continue
                rhs = rhs + src_field[r3, r2, r1]

    if no_store:
        assignments = [
            ps.Assignment(v, rhs * 0.25),
            psast.Conditional(
                sp.Eq(v, 2.2),
                psast.Block([ps.Assignment(dst_field[0, 0, 0], v)]),
            ),
        ]
    else:
        assignments = [ps.Assignment(dst_field[0, 0, 0], rhs * 0.25)]
    if blocking_factors != (1, 1, 1):
        return blockThreads(assignments, blocking_factors)
    else:
        return assignments


def getStencilKernel(
    stencil_range, stencil_type, block_size, blocking_factors, fieldSize, datatype
):
    domainSize = [s - 2 * stencil_range for s in fieldSize]

    dst_field = ps.fields(
        "dst : " + datatype + "[{}, {}, {}]".format(*fieldSize),
        layout="reverse_numpy",
    )
    src_field = ps.fields(
        "src : " + datatype + "[{}, {}, {}]".format(*fieldSize), layout="reverse_numpy"
    )
    ghost_layers = stencil_range
    if datatype == "float32":
        byte_offset = ghost_layers * 4
    else:
        byte_offset = ghost_layers * 8

    dst_field.byte_offset = byte_offset
    src_field.byte_offset = byte_offset

    ast = ps.create_kernel(
        getStarAssignments(src_field, dst_field, stencil_range, blocking_factors)
        if stencil_type == "star"
        else getBoxAssignments(src_field, dst_field, stencil_range, blocking_factors),
        target="gpu",
        data_type={"dst": "float", "src": "float", "_constant": "float"},
        ghost_layers=ghost_layers,
        gpu_indexing_params={
            "block_size": block_size,
            "permute_block_size_dependent_on_layout": False,
            "blocking_factors": blocking_factors,
        },
    )

    bufferSizeBytes = (
        fieldSize[0] * fieldSize[1] * fieldSize[2] * (4 if datatype == "float32" else 8)
    )

    return ast, domainSize, bufferSizeBytes, byte_offset
