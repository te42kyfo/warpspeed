#!/usr/bin/env python3

import numpy as np

from functools import partial, reduce
from operator import mul

import predict
from volumes_isl import getMemLoadBlockVolumeISL

from collections import namedtuple


class printingVisitor:
    def count(self, addresses, multiplicity=1, datatype=8):
        print(addresses)


class noneVisitor:
    def count(self, addresses):
        pass


class CL32Visitor:
    def __init__(self):
        self.CLs = 0

    def count(self, addresses, multiplicity=1, datatype=8):
        self.CLs += np.unique(addresses // 32).size * multiplicity


class CL64Visitor:
    def __init__(self):
        self.CLs = 0

    def count(self, addresses, multiplicity=1, datatype=8):
        self.CLs += np.unique(addresses // 64).size * multiplicity


class CL128Visitor:
    def __init__(self):
        self.CLs = 0

    def count(self, addresses, multiplicity=1, datatype=8):
        self.CLs += np.unique(addresses // 128).size * multiplicity


class PageVisitor:
    def __init__(self, pageSize):
        self.pageSize = pageSize
        self.pages = 0

    def count(self, addresses, multiplicity=1, datatype=8):
        self.pages += np.unique(addresses // (self.pageSize)).size * multiplicity


def countBankConflicts(laneAddresses, datatype=8):
    addresses = list(set([l // datatype for l in laneAddresses]))

    banks = [0] * (128 // datatype)
    maxCycles = 0
    for a in addresses:
        bank = int(a) % (128 // datatype)
        banks[bank] += 1
        maxCycles = max(maxCycles, banks[bank])
    return maxCycles


class L1thruVisitorNV:
    def __init__(self):
        self.cycles = 0
        self.dataPipeCycles = 0
        self.tagCycles = 0
        self.uniform = 0
        self.coalesced = 0

    def count(self, laneAddresses, multiplicity=1, datatype=8):
        if len(set(laneAddresses)) == 1:  # uniform loads
            self.uniform += multiplicity

        coalesced = True
        for i in range(len(laneAddresses)):
            if laneAddresses[i] - laneAddresses[0] != i * 4:
                coalesced = False
        if coalesced:
            self.coalesced += multiplicity

        self.dataPipeCycles += (
            countBankConflicts(laneAddresses, datatype) + 0.25
        ) * multiplicity

        self.tagCycles += (
            max(1, np.unique(laneAddresses // 128).size) + 2
        ) * multiplicity


class L1thruVisitorCDNA:
    def __init__(self):
        self.cycles = 0

    def count(self, laneAddresses, multiplicity=1, datatype=8):
        addresses = list(set([l // 8 for l in laneAddresses]))
        banks = [0] * 16
        maxCycles = 0
        for a in addresses:
            bank = int(a) % 16
            banks[bank] += 1
            maxCycles = max(maxCycles, banks[bank])
        self.cycles += (
            max(4 * maxCycles, np.unique(laneAddresses // 1024).size)
        ) * multiplicity


class DummyFieldAccess:
    def __init__(self, address, datatype, multiplicity):
        self.linearAddresses = [address]
        self.datatype = datatype
        self.multiplicity = multiplicity


def gridIteration(fields, innerSize, outerSize, visitor):
    idx = np.arange(0, innerSize[0], dtype=np.int32)
    idy = np.arange(0, innerSize[1], dtype=np.int32)
    idz = np.arange(0, innerSize[2], dtype=np.int32)
    x, y, z = np.meshgrid(idx, idy, idz)

    for field in fields:
        for outerId in np.ndindex(outerSize):
            addresses = np.empty(0)
            for addressLambda in field.linearAddresses:
                addresses = np.concatenate(
                    (
                        addresses,
                        np.asarray(
                            addressLambda(x, y, z, *outerId, *innerSize)
                        ).ravel(),
                    )
                )
            visitor.count(addresses, field.multiplicity, field.datatype)


def getWarp(warpSize, block):
    warp = (
        min(warpSize, block[0]),
        min(block[1], max(1, warpSize // block[0])),
        min(block[2], max(1, warpSize // block[0] // block[1])),
    )
    return warp


def getL1Cycles(block, grid, loadStoreFields, L1Model):
    separatedFields = [
        DummyFieldAccess(a, field.datatype, field.multiplicity)
        for field in loadStoreFields
        for a in field.linearAddresses
    ]

    fieldCycles = {}

    L1Components = namedtuple("L1Components", "dataPipeCycles tagCycles total")

    if L1Model == "CDNA":
        halfWarp = getWarp(16, block)
        outerSize = tuple(grid[i] * block[i] // halfWarp[i] for i in range(0, 3))
        visitor = L1thruVisitorCDNA()
        gridIteration(separatedFields, halfWarp, outerSize, visitor)

        return (visitor.cycles) / outerSize[0] / outerSize[1] / outerSize[2] * 2
    else:
        dataPipeCycles = 0
        tagCycles = 0
        totalCycles = 0
        for field in loadStoreFields:
            separatedFields = [
                DummyFieldAccess(a, field.datatype, field.multiplicity)
                for a in field.linearAddresses
            ]

            visitor = L1thruVisitorNV()
            warp = getWarp(32, block)
            outerSize = tuple(grid[i] * block[i] // warp[i] for i in range(0, 3))
            gridIteration(separatedFields, warp, outerSize, visitor)

            fieldDataPipeCycles = (
                visitor.dataPipeCycles / outerSize[0] / outerSize[1] / outerSize[2] + 4
            )
            fieldTagCycles = (
                visitor.tagCycles / outerSize[0] / outerSize[1] / outerSize[2] / 4
            )

            fieldCycles[field.name] = L1Components(
                fieldDataPipeCycles,
                fieldTagCycles,
                max(fieldDataPipeCycles, fieldTagCycles),
            )

            dataPipeCycles += fieldDataPipeCycles
            tagCycles += fieldTagCycles
            totalCycles = max(fieldDataPipeCycles, fieldTagCycles)
        # print(visitor.cycles / outerSize[0] / outerSize[1] / outerSize[2])
        # print(visitor.tagCycles / outerSize[0] / outerSize[1] / outerSize[2])
        # print(visitor.dataPipeCycles / outerSize[0] / outerSize[1] / outerSize[2])
        # print(visitor.uniform / outerSize[0] / outerSize[1] / outerSize[2])
        # print(visitor.coalesced / outerSize[0] / outerSize[1] / outerSize[2])

        fieldCycles["total"] = L1Components(dataPipeCycles, tagCycles, totalCycles)
        return fieldCycles


def getL1AllocatedLoadBlockVolume(block, grid, loadAddresses, CLAllocationSize):
    visitor = CL128Visitor()
    gridIteration(loadAddresses, block, grid, visitor)
    return visitor.CLs * CLAllocationSize / grid[0] / grid[1] / grid[2]


def getL1WarpLoadVolume(block, loadAddresses, fetchSize):
    warp = getWarp(32, block)
    grid = tuple(block[i] // warp[i] for i in range(0, 3))
    visitor = CL32Visitor()
    gridIteration(loadAddresses, warp, grid, visitor)
    return visitor.CLs * fetchSize


def getL1TLBPages(block, grid, addresses, pageSize):
    visitor = PageVisitor(pageSize)

    gridIteration(addresses, block, grid, visitor)

    return visitor.pages / grid[0] / grid[1] / grid[2]


def getL2LoadBlockVolume(block, grid, loadAddresses, fetchSize):
    if fetchSize == 32:
        visitor = CL32Visitor()
    elif fetchSize == 64:
        visitor = CL64Visitor()

    gridIteration(loadAddresses, block, grid, visitor)
    return visitor.CLs * fetchSize / grid[0] / grid[1] / grid[2]


def getL2StoreBlockVolume(block, grid, storeFields, fetchSize):
    warp = getWarp(32, block)

    outerSize = tuple(grid[i] * block[i] // warp[i] for i in range(0, 3))

    separatedFields = [
        DummyFieldAccess(a, field.datatype, field.multiplicity)
        for field in storeFields
        for a in field.linearAddresses
    ]

    if fetchSize == 32:
        visitor = CL32Visitor()
    elif fetchSize == 64:
        visitor = CL64Visitor()
    gridIteration(separatedFields, warp, outerSize, visitor)
    return visitor.CLs * fetchSize / grid[0] / grid[1] / grid[2]


def getMemLoadWaveVolume(block, grid, loadAddresses):
    visitor = CL32Visitor()

    innerSize = tuple(block[i] * grid[i] for i in range(3))
    gridIteration(loadAddresses, innerSize, (1, 1, 1), visitor)
    return visitor.CLs * 32


def getMemStoreWaveVolume(block, grid, addresses):
    visitor = CL32Visitor()

    innerSize = tuple(block[i] * grid[i] for i in range(3))

    gridIteration(addresses, innerSize, (1, 1, 1), visitor)
    return visitor.CLs * 32


def getWaveLoadTLBPages(block, grid, addresses, pageSize):
    visitor = PageVisitor(pageSize)

    innerSize = tuple(block[i] * grid[i] for i in range(3))

    gridIteration(addresses, innerSize, (1, 1, 1), visitor)
    return visitor.pages
