#!/usr/bin/env python3

import numpy as np


class printingVisitor:
    def count(self, addresses):
        print(addresses)


class noneVisitor:
    def count(self, addresses):
        pass


class CL32Visitor:
    def __init__(self):
        self.CLs = 0

    def count(self, addresses):
        self.CLs += np.unique( addresses // 4 ).size


class L1thruVisitor:
    def __init__(self):
        self.cycles = 0

    def count(self, addresses):
        addresses = list(set(addresses))
        banks = [0] * 16
        maxCycles = 0
        for a in addresses:
            bank = int(a) % 16
            banks[bank] += 1
            maxCycles = max(maxCycles, banks[bank])
        self.cycles += maxCycles


def gridIteration(fields, innerSize, outerSize, visitor):

    idx = np.arange(0, innerSize[0], dtype=np.int32)
    idy = np.arange(0, innerSize[1], dtype=np.int32)
    idz = np.arange(0, innerSize[2], dtype=np.int32)
    x, y, z = np.meshgrid(idx, idy, idz)

    for field in fields:
        for outerId in np.ndindex(outerSize):
            addresses = np.empty(0)
            for addressLambda in fields[field]:
                addresses = np.concatenate(( addresses, addressLambda(x, y, z, *outerId, *innerSize).ravel()))
            visitor.count(addresses)


def getL2LoadBlockVolume(block, grid, loadAddresses):
    visitor = CL32Visitor()
    gridIteration(loadAddresses, block, grid, visitor)
    return visitor.CLs * 32 / grid[0] / grid[1] / grid[2]


def getL2StoreBlockVolume(block, grid, storeAddresses):
    warp = (
        min(32, block[0]),
        min(block[1], max(1, 32 // block[0])),
        min(block[2], max(1, 32 // block[0] // block[1])),
    )

    outerSize = tuple(grid[i] * block[i] // warp[i] for i in range(0,3))

    visitor = CL32Visitor()
    gridIteration(storeAddresses, warp, outerSize, visitor)
    return visitor.CLs * 32 / grid[0] / grid[1] / grid[2]

def getL1Cycles(block, grid, loadStoreAddresses):

    halfWarp = (
        min(16, block[0]),
        min(block[1], max(1, 16 // block[0])),
        min(block[2], max(1, 16 // block[0] // block[1])),
    )

    outerSize = tuple(grid[i] * block[i] // halfWarp[i] for i in range(0,3))

    visitor = L1thruVisitor()
    gridIteration(loadStoreAddresses, halfWarp, outerSize, visitor)
    return visitor.cycles / outerSize[0] / outerSize[1] / outerSize[2] * 2

def getMemLoadBlockVolume(block, grid, loadAddresses):
    visitor = CL32Visitor()

    innerSize = tuple(block[i] * grid[i] for i in range(3))

    gridIteration(loadAddresses, innerSize, (1,1,1), visitor)
    return visitor.CLs * 32 / grid[0] / grid[1] / grid[2]


def getMemStoreBlockVolume(block, grid, loadAddresses):
    visitor = CL32Visitor()

    innerSize = tuple(block[i] * grid[i] for i in range(3))

    gridIteration(loadAddresses, innerSize, (1,1,1), visitor)
    return visitor.CLs * 32 / grid[0] / grid[1] / grid[2]
