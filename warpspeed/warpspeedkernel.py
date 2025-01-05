#!/usr/bin/env python3

from predict_metrics import *
import sympy as sp
import math


class Field:
    def __init__(
        self,
        name,
        addresses,
        datatype,
        dimensions,
        alignment,
        multiplicity=1,
        scalar=False,
    ):
        self.name = name
        addresses = [[a] if isinstance(a, str) else a for a in addresses]
        self.NDAddresses = [[str(sp.sympify(expr)) for expr in a] for a in addresses]
        # Extend addresses to 3D with "0" as address expressions
        self.NDAddresses = [
            (
                tuple(list(a) + ["0"] * (3 - len(a)))
                if not isinstance(a, str)
                else (a, "0", "0")
            )
            for a in self.NDAddresses
        ]
        self.NDAddresses = list(set(self.NDAddresses))

        self.datatype = datatype
        self.dimensions = dimensions
        self.alignment = alignment

        self.multiplicity = multiplicity
        self.scalar = scalar

        def linearizeExpr(expr3D):
            exprString = "({5} + {0} + ({1}) * {3} + ({2}) * {4}) * {6}".format(
                *expr3D,
                self.dimensions[0],
                self.dimensions[1] * self.dimensions[0],
                self.alignment,
                self.datatype
            )
            linExpr = sp.sympify(exprString)
            linExpr = linExpr.subs(
                [
                    ("tidx", "blockIdx * blockDimx + tidx"),
                    ("tidy", "blockIdy * blockDimy + tidy"),
                    ("tidz", "blockIdz * blockDimz + tidz"),
                ]
            )
            return linExpr

        def lambdifyExpr(linExpr):
            return sp.lambdify(
                sp.symbols(
                    "tidx, tidy, tidz, blockIdx, blockIdy, blockIdz, blockDimx, blockDimy, blockDimz"
                ),
                linExpr,
            )

        self.datatypes = [datatype] * len(addresses)
        self.linearExpressions = [linearizeExpr(a) for a in self.NDAddresses]
        self.linearAddresses = [lambdifyExpr(a) for a in self.linearExpressions]


def fuseAccesses(fields):
    for f in fields:
        accesses = list(
            [list(a) for a in zip(f.linearExpressions, f.linearAddresses, f.datatypes)]
        )

        changed = True
        while changed:
            changed = False
            accesses.sort(key=lambda a: -a[2])
            for c, o in [
                (c, o) for c in range(len(accesses)) for o in range(len(accesses))
            ]:
                if o >= len(accesses):
                    continue

                diff = accesses[o][0] - accesses[c][0]
                if (
                    diff == sp.sympify(accesses[c][2])
                    and accesses[c][2] + accesses[o][2] <= 16
                ):
                    accesses[c][2] += accesses[o][2]
                    del accesses[o]
                    changed = True
                    break

        f.linearExpressions = [a[0] for a in accesses]
        f.linearAddresses = [a[1] for a in accesses]
        f.datatypes = [a[2] for a in accesses]


class WarpspeedKernel:
    def fuseAccesses(self):
        fuseAccesses(self.loadFields)
        fuseAccesses(self.storeFields)

    def __init__(self, loadFields, storeFields, registers, flops=0, flins=0, fp_type=8):
        self.registers = registers

        if flins == 0:
            self.flins = int(math.ceil(flops / 2))

        self.flops = flops

        self.loadFields = loadFields
        self.storeFields = storeFields
        self.fp_type = fp_type
        self.flops = flops
        self.flins = flins
