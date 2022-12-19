#!/usr/bin/env python3

import pystencils as ps
import pystencils.astnodes as psast
import sympy as sp
import functools
import numpy as np
from pystencils.simp import sympy_cse_on_assignment_list
import random
from sympy import Symbol, numbered_symbols

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

    #random.shuffle(new_assignments)

    for a in assignments:
        new_assignments.append(ps.Assignment(a.lhs, a.rhs.subs( zip(loads.keys(), loads.values())  ) ))

    #for a in new_assignments:
    #    print(a)
    return new_assignments




def blockThreads(assignments, blocking_factors):
    blockedAssignments = []
    for bx,by,bz in np.ndindex(blocking_factors):
        def shiftIndices(expr):
            if isinstance(expr, psast.Field.Access):
                return expr.get_shifted(bx,by,bz)
            if len(expr.args) > 0:
                return expr.func(*[ shiftIndices(arg) for arg in expr.args ])
            return expr

        for a in assignments:
            blockedAssignments.append(shiftIndices(a))

    return orderLoads(sympy_cse_on_assignment_list(blockedAssignments))

class PS3DStencil:
    def __init__(self, size, ghost_layers):
        self.size = [s - 2 * ghost_layers for s in size]
        self.ghost_layers = ghost_layers

        self.dst_field = ps.fields("dst : double[{}, {}, {}]".format(*size), layout="reverse_numpy")
        self.src_field = ps.fields("src : double[{}, {}, {}]".format(*size), layout="reverse_numpy")

        byte_offset = ghost_layers * np.dtype('d').itemsize
        self.dst_field.byte_offset = byte_offset
        self.src_field.byte_offset = byte_offset
        self.dh = None

    def getStarAssignments(self, stencil_range, blocking_factors=(1,1,1), no_store=False):
        assignments = []
        v = sp.Symbol("v")
        if stencil_range > -1:
            rhs = self.src_field[0, 0, 0]
        else:
            rhs = 0.25
        for r in range(1, stencil_range + 1):
            rhs = (
                rhs
                + self.src_field[0, 0, r]
                + self.src_field[0, 0, -r]
                + self.src_field[0, r, 0]
                + self.src_field[0, -r, 0]
                + self.src_field[r, 0, 0]
                + self.src_field[-r, 0, 0]
            )

        if no_store:
            assignments = [
                ps.Assignment(v, rhs * 0.25),
                psast.Conditional(
                    sp.Eq(v, 2.2),
                    psast.Block([ps.Assignment(self.dst_field[0, 0, 0], v)]),
                ),
            ]
        else:
            assignments = [ps.Assignment(self.dst_field[0, 0, 0], rhs * 0.25)]
        if blocking_factors != (1,1,1):
            return blockThreads(assignments, blocking_factors)
        else:
            return assignments

    def getStarKernel(self, block_size, stencil_range, blocking_factors=(1,1,1)):
        ast = ps.create_kernel(
            self.getStarAssignments(stencil_range, blocking_factors),
            target="gpu",
            ghost_layers = self.ghost_layers,
            gpu_indexing_params={
                "block_size": block_size,
                "permute_block_size_dependent_on_layout": False,
                "blocking_factors" : blocking_factors
            },
        )

        kernel = ast.compile()
        return kernel

    def getBoxAssignments(self, stencil_range, blocking_factors=(1,1,1), no_store=False):
        assignments = []
        v = sp.Symbol("v")
        if stencil_range > -1:
            rhs = self.src_field[0, 0, 0]
        else:
            rhs = 0.25


        for r1 in range(-stencil_range, stencil_range + 1):
            for r2 in range(-stencil_range, stencil_range + 1):
                for r3 in range(-stencil_range, stencil_range + 1):
                    if r1 == 0 and r2 == 0 and r3 == 0:
                        continue
                    rhs = (
                        rhs
                        + self.src_field[r3, r2, r1]
                    )

        if no_store:
            assignments = [
                ps.Assignment(v, rhs * 0.25),
                psast.Conditional(
                    sp.Eq(v, 2.2),
                    psast.Block([ps.Assignment(self.dst_field[0, 0, 0], v)]),
                ),
            ]
        else:
            assignments = [ps.Assignment(self.dst_field[0, 0, 0], rhs * 0.25)]
        if blocking_factors != (1,1,1):
            return blockThreads(assignments, blocking_factors)
        else:
            return assignments

    def getBoxKernel(self, block_size, stencil_range, blocking_factors=(1,1,1)):
        ast = ps.create_kernel(
            self.getBoxAssignments(stencil_range, blocking_factors),
            target="gpu",
            ghost_layers = self.ghost_layers,
            gpu_indexing_params={
                "block_size": block_size,
                "permute_block_size_dependent_on_layout": False,
                "blocking_factors" : blocking_factors
            },
        )

        kernel = ast.compile()
        return kernel
    def getRunFunc(self, kernel):

        if self.sh is None:
            self.dh = ps.create_data_handling(
                (self.size[0], self.size[1], self.size[2]), default_target="gpu"
            )

            self.gpu_dst_field = self.dh.add_array(
                "dst", values_per_cell=1, ghost_layers=ghost_layers
            )
            self.dh.fill("dst", 0.0, ghost_layers=True)
            self.gpu_src_field = self.dh.add_array(
                "src", values_per_cell=1, ghost_layers=ghost_layers
            )
            self.dh.fill("src", 0.0, ghost_layers=True)
            self.dh.all_to_gpu()

        return functools.partial(self.dh.run_kernel, kernel, self.gpu_dst_field, self.gpu_src_field)


class PS2DStencil:
    def __init__(self, size, ghost_layers):
        self.size = [s - 2 * ghost_layers for s in size]
        self.ghost_layers = ghost_layers
        self.dh = ps.create_data_handling(
            (self.size[0], self.size[1]), default_target="gpu"
        )

        self.dst_field = self.dh.add_array(
            "dst", values_per_cell=1, ghost_layers=ghost_layers
        )
        # self.dh.fill("dst", 0.0, ghost_layers=False)
        self.src_field = self.dh.add_array(
            "src", values_per_cell=1, ghost_layers=ghost_layers
        )
        # self.dh.fill("src", 0.0, ghost_layers=False)
        self.dh.all_to_gpu()

    def getStarAssignments(self, stencil_range, no_store=False):
        v = sp.Symbol("v")
        if stencil_range > -1:
            rhs = self.src_field[0, 0]
        else:
            rhs = 0.25
        for r in range(1, stencil_range + 1):
            rhs = (
                rhs
                + self.src_field[0, r]
                + self.src_field[0, -r]
                + self.src_field[r, 0]
                + self.src_field[-r, 0]
            )
        assignments = []
        if no_store:
            assignments = [
                ps.Assignment(v, rhs * 0.25),
                psast.Conditional(
                    sp.Eq(v, 2.2),
                    psast.Block([ps.Assignment(self.dst_field[stencil_range, 0], v)]),
                ),
            ]
        else:
            assignments = [ps.Assignment(self.dst_field[0, 0], rhs * 0.25)]

        if blocking_factors != (1,1,1):
            return blockThreads(assignments, blocking_factors)
        else:
            return assignments

    def getStarKernel(self, block_size, stencil_range, blocking_factors=(1,1,1)):
        ast = ps.create_kernel(
            self.getStarAssignments(stencil_range, blocking_factors),
            target="gpu",
            ghost_layers = self.ghost_layers,
            gpu_indexing_params={
                "block_size": block_size,
                "permute_block_size_dependent_on_layout": False,
                "blocking_factors": blocking_factors
            },
        )

        kernel = ast.compile()
        return kernel

    def getRunFunc(self, kernel):
        return functools.partial(self.dh.run_kernel, kernel)
