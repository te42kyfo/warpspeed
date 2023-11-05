#!/usr/bin/env python3

from predict_metrics import *
import sympy as sp


class Field:
    def __init__(
        self, name, addresses, datatype, dimensions, alignment, multiplicity=1
    ):
        self.name = name
        self.NDAddresses = [[str(sp.sympify(expr)) for expr in a] for a in addresses]
        # Extend addresses to 3D with "0" as address expressions
        self.NDAddresses = [
            tuple(list(a) + ["0"] * (3 - len(a)))
            if not isinstance(a, str)
            else (a, "0", "0")
            for a in self.NDAddresses
        ]
        self.NDAddresses = list(set(self.NDAddresses))

        self.datatype = datatype
        self.dimensions = dimensions
        self.alignment = alignment

        self.multiplicity = multiplicity

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

        self.linearExpressions = [linearizeExpr(a) for a in self.NDAddresses]
        self.linearAddresses = [lambdifyExpr(a) for a in self.linearExpressions]


class WarpspeedKernel:
    def __init__(self, loadFields, storeFields, registers, flops=0):
        self.registers = registers
        self.flops = flops

        self.loadFields = loadFields
        self.storeFields = storeFields

        self.flops = flops
