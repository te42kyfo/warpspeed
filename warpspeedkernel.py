#!/usr/bin/env python3

from predict_metrics import *
import sympy as sp




class WarpspeedKernel:
    def __init__(self, loadExprs, storeExprs, domain, registers):
        self.registers = registers
        self.loadExprs3D = {field : [(expr, "0", "0") for expr in loadExprs[field]] for field in loadExprs}
        self.storeExprs3D = {field : [(expr, "0", "0") for expr in storeExprs[field]] for field in storeExprs}

        def sympifyExprs(exprs):
            return {field : [sp.lambdify( sp.symbols("tidx, tidy, tidz, blockIdx, blockIdy, blockIdz, blockDimx, blockDimy, blockDimz"),
                                sp.sympify( expr)) for expr in exprs[field]] for field in exprs}

        self.loadExprs = sympifyExprs(loadExprs)
        self.storeExprs = sympifyExprs(storeExprs)

    def getLoadExprs3D(self):
        return self.loadExprs3D

    def getStoreExprs3D(self):
        return self.storeExprs3D

    def genLoads(self):
        return self.loadExprs

    def genStores(self):
        return self.storeExprs


class WarpspeedGridKernel:
    def __init__(self, loadExprs3D, storeExprs3D, domain, registers, alignment):
        self.registers = registers
        self.loadExprs3D = loadExprs3D
        self.storeExprs3D = storeExprs3D

        def linearizeExpr(expr3D):
            return sp.lambdify( sp.symbols("tidx, tidy, tidz, blockIdx, blockIdy, blockIdz, blockDimx, blockDimy, blockDimz"),
                                sp.sympify( "{5} + ({0}) + ({1}) * {3} + ({2}) * {4}".format( *expr3D, domain[0], domain[1]*domain[0], alignment)))
        def linearizeFields(exprs):
            return {field : [ linearizeExpr(expr) for expr in exprs[field]  ] for field in exprs.keys()}

        self.loadExprs = linearizeFields(loadExprs3D)
        self.storeExprs = linearizeFields(storeExprs3D)

    def getLoadExprs3D(self):
        return self.loadExprs3D

    def getStoreExprs3D(self):
        return self.storeExprs3D

    def genLoads(self):
        return self.loadExprs

    def genStores(self):
        return self.storeExprs
