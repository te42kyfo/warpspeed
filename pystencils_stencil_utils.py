#!/usr/bin/env python3

import pystencils as ps
import pystencils.astnodes as psast
import sympy as sp
import functools


class PS3DStencil:
    def __init__(self, size):
        self.size = size

        self.dh = ps.create_data_handling(
            (size[0], size[1], size[2]), default_target="gpu"
        )

        self.dst_field = self.dh.add_array("dst", values_per_cell=1)
        # self.dh.fill("dst", 0.0, ghost_layers=False)
        self.src_field = self.dh.add_array("src", values_per_cell=1)
        # self.dh.fill("src", 0.0, ghost_layers=False)
        self.dh.all_to_gpu()

    def getStarAssignments(self, stencil_range, no_store=False):
        v = sp.Symbol("v")
        if stencil_range > 0:
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
            return [
                ps.Assignment(v, rhs * 0.25),
                psast.Conditional(
                    sp.Eq(v, 2.2),
                    psast.Block(
                        [ps.Assignment(self.dst_field[stencil_range, 0, 0], v)]
                    ),
                ),
            ]
        else:
            return [ps.Assignment(self.dst_field[stencil_range, 0, 0], rhs * 0.25)]

    def getStarKernel(self, block_size, stencil_range):
        ast = ps.create_kernel(
            self.getStarAssignments(stencil_range),
            target="gpu",
            gpu_indexing_params={
                "block_size": block_size,
                "permute_block_size_dependent_on_layout": False,
            },
        )

        kernel = ast.compile()
        return kernel

    def getRunFunc(self, kernel):
        return functools.partial(self.dh.run_kernel, kernel)
