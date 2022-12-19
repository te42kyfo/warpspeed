#!/usr/bin/env python3

from pystencils.sympyextensions import count_operations, count_operations_in_ast
import pystencils as ps
import sympy as sp
from pystencils.astnodes import (
    KernelFunction,
    LoopOverCoordinate,
    ResolvedFieldAccess,
    SympyAssignment,
    Conditional,
)


def search_resolved_field_accesses_in_ast(ast):
    def visit(node, reads, writes):
        if isinstance(node, Conditional):
            return
        if not isinstance(node, SympyAssignment):
            for a in node.args:
                visit(a, reads, writes)
            return

        for expr, accesses in [(node.lhs, writes), (node.rhs, reads)]:
            accesses.update(expr.atoms(ResolvedFieldAccess))

    read_accesses = set()
    write_accesses = set()
    for eq in ast.assignments:
        visit(eq, read_accesses, write_accesses)
    return read_accesses, write_accesses

def getFieldExprs(kernel):

    reads, writes = search_resolved_field_accesses_in_ast(kernel)

    def getFieldsDict(accesses):
        fields = dict()
        addresses = []
        for access in accesses:
            expr = access.indices[0]

            for a in kernel.assignments:
                if not isinstance(a, SympyAssignment):
                    continue
                if a.lhs == access.base.label:
                    expr = expr + a.rhs

            for a in kernel.assignments:
                if not isinstance(a, SympyAssignment):
                    continue
                if a.lhs in expr.atoms():
                    expr = expr.subs(a.lhs, a.rhs)

            dim = kernel.indexing._dim

            # substitutions = zip(ps.gpucuda.indexing.BLOCK_DIM[:dim], kernel.indexing._block_size)
            # substitutedAddress = expr.subs( substitutions)
            substitutedAddress = expr

            if (
                len(substitutedAddress.atoms(ps.kernelparameters.FieldPointerSymbol))
                == 0
            ):
                continue
            field = substitutedAddress.atoms(
                ps.kernelparameters.FieldPointerSymbol
            ).pop()
            substitutedAddress = substitutedAddress.subs(
                zip(
                    substitutedAddress.atoms(ps.kernelparameters.FieldPointerSymbol),
                    [32 - access.field.byte_offset // 8],
                )
            )

            if field not in fields:
                fields[field] = []
            fields[field].append(substitutedAddress)
        return fields

    storeFields = getFieldsDict(writes)
    loadFields = getFieldsDict(reads)

    return storeFields, loadFields

def getFieldExprs3D(ast):

    reads, writes = search_resolved_field_accesses_in_ast(ast)

    def getFieldsDict(accesses):
        fields = dict()
        addresses = []
        for access in accesses:

            field = str(access.field)
            if len(access.idx_coordinate_values) > 0:
               field += "_" + str(access.idx_coordinate_values[0])
            if field not in fields:
                fields[field] = []
            fields[field].append(("tidx * {} + {}".format(ast.indexing._blocking_factors[0], access.offsets[0]),
                                  "tidy * {} + {}".format(ast.indexing._blocking_factors[1], access.offsets[1]),
                                  "tidz * {} + {}".format(ast.indexing._blocking_factors[2], access.offsets[2])))
        return fields

    return getFieldsDict(reads), getFieldsDict(writes)


def lambdifyExprs(fields):
    fieldsLambdas = dict()
    for field in fields:
        fieldsLambdas[field] = []
        for expr in fields[field]:
            addressLambda = sp.lambdify(
                (
                    ps.gpucuda.indexing.THREAD_IDX
                    + ps.gpucuda.indexing.BLOCK_IDX
                    + ps.gpucuda.indexing.BLOCK_DIM
                ),
                expr,
            )
            fieldsLambdas[field].append(addressLambda)
    return fieldsLambdas


def simplifyExprs(fields):
    newFields = dict()
    for field in fields:
        newExprs = []
        for expr in fields[field]:
            newExprs.append(
                expr.subs(
                    zip(
                        ps.gpucuda.indexing.BLOCK_IDX + ps.gpucuda.indexing.BLOCK_DIM,
                        [0] * 6,
                    ),
                ).subs(
                    zip(ps.gpucuda.indexing.THREAD_IDX, sp.symbols("tidx tidy tidz"))
                )
            )
        newFields[field] = newExprs
    return newFields


class PystencilsWarpSpeedField:
    def __init__(self, name, linearAddresses, NDAddresses, datatype, dimensions, alignment):
        self.name = name
        self.linearAddresses = linearAddresses
        self.NDAddresses = NDAddresses
        self.datatype = datatype
        self.dimensions = dimensions
        self.alignment = alignment

class PyStencilsWarpSpeedKernel:
    def __init__(self, ast):
        self.storeExprs, self.loadExprs = getFieldExprs(ast)

        print(self.loadExprs)

        self.storeFields = lambdifyExprs(self.storeExprs)
        self.loadFields = lambdifyExprs(self.loadExprs)
        #self.storeExprs = simplifyExprs(self.storeExprs)
        #self.loadExprs = simplifyExprs(self.loadExprs)

        self.loadExprs3D, self.storeExprs3D = getFieldExprs3D(ast)

        print(self.loadExprs3D)
        print()

        operation_count = count_operations_in_ast(ast)

        self.flops = operation_count["adds"] + operation_count["muls"]

        self.ast = ast

    def getLoadExprs3D(self):
        return self.loadExprs3D

    def getStoreExprs3D(self):
        return self.storeExprs3D



    def genLoads(self):
        return self.loadFields

    def genStores(self):
        return self.storeFields

    #def genLoadExprs(self):
    #    return self.loadExprs

    #def genStoreExprs(self):
    #    return self.storeExprs
