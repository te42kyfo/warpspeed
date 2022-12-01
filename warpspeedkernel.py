#!/usr/bin/env python3

from predict_metrics import *
import sympy as sp


class Field:
    def __init__(self, name, addresses, datatype, dimensions, alignment):
        self.name = name
        self.NDAddresses = addresses

        self.datatype = datatype
        self.dimensions = dimensions
        self.alignment = alignment

        def linearizeExpr(expr3D):
            exprString = "({5} + {0} + ({1}) * {3} + ({2}) * {4}) * {6}".format(
                *expr3D,
                self.dimensions[0],
                self.dimensions[1] * self.dimensions[0],
                self.alignment,
                self.datatype
            )
            linExpr = sp.sympify(exprString)
            return sp.lambdify(
                sp.symbols(
                    "tidx, tidy, tidz, blockIdx, blockIdy, blockIdz, blockDimx, blockDimy, blockDimz"
                ),
                linExpr,
            )

        # Extend addresses to 3D with "0" as address expressions
        self.NDAddresses = [
            tuple(list(a) + ["0"] * (3 - len(a)))
            if not isinstance(a, str)
            else (a, "0", "0")
            for a in self.NDAddresses
        ]

        self.linearAddresses = [linearizeExpr(a) for a in self.NDAddresses]


class WarpspeedKernel:
    def __init__(self, loadFields, storeFields, registers, flops=0):
        self.registers = registers
        self.flops = flops

        self.loadFields = loadFields
        self.storeFields = storeFields

        self.flops = flops

    def getLoadExprs3D(self):
        return self.loadExprs3D

    def getStoreExprs3D(self):
        return self.storeExprs3D

    def genLoads(self):
        return self.loadExprs

    def genStores(self):
        return self.storeExprs
