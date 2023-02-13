#!/usr/bin/env python3

import numpy as np

from functools import partial, reduce
from operator import mul

import predict
from volumes_isl import getMemLoadBlockVolumeISL


class printingVisitor:
    def count(self, addresses):
        print(addresses)


class noneVisitor:
    def count(self, addresses):
        pass


class CL32Visitor:
    def __init__(self):
        self.CLs = 0

    def count(self, addresses, multiplicity=1):
        self.CLs += np.unique(addresses // 32).size * multiplicity


class CL64Visitor:
    def __init__(self):
        self.CLs = 0

    def count(self, addresses, multiplicity=1):
        self.CLs += np.unique(addresses // 64).size * multiplicity


class CL128Visitor:
    def __init__(self):
        self.CLs = 0

    def count(self, addresses, multiplicity=1):
        self.CLs += np.unique(addresses // 128).size * multiplicity


class PageVisitor:
    def __init__(self, pageSize):
        self.pageSize = pageSize
        self.pages = 0

    def count(self, addresses, multiplicity=1):
        self.pages += np.unique(addresses // (self.pageSize)).size * multiplicity

class L1thruVisitorNV:
    def __init__(self):
        self.cycles = 0

    def count(self, laneAddresses, multiplicity=1):
        addresses = list(set([l // 8 for l in laneAddresses]))
        banks = [0] * 16
        maxCycles = 0
        for a in addresses:
            bank = int(a) % 16
            banks[bank] += 1
            maxCycles = max(maxCycles, banks[bank])
        self.cycles += max(maxCycles,  np.unique(laneAddresses // 1024).size)*multiplicity

class L1thruVisitorCDNA:
    def __init__(self):
        self.cycles = 0

    def count(self, laneAddresses, multiplicity=1):
        addresses = list(set([l // 8 for l in laneAddresses]))
        banks = [0] * 16
        maxCycles = 0
        for a in addresses:
            bank = int(a) % 16
            banks[bank] += 1
            maxCycles = max(maxCycles, banks[bank])
        self.cycles += max(4, maxCycles,  np.unique(laneAddresses // 1024).size)*multiplicity

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
                    (addresses, np.asarray(addressLambda(x, y, z, *outerId, *innerSize)).ravel())
                )
            visitor.count(addresses, field.multiplicity)


def getWarp(warpSize, block):
    warp = (
        min(warpSize, block[0]),
        min(block[1], max(1, warpSize // block[0])),
        min(block[2], max(1, warpSize // block[0] // block[1])),
    )
    return warp


def getL1Cycles(block, grid, loadStoreFields, L1Model):

    halfWarp = getWarp(16, block)
    outerSize = tuple(grid[i] * block[i] // halfWarp[i] for i in range(0, 3))
    if L1Model=="CDNA":
        visitor = L1thruVisitorCDNA()
    else:
        visitor = L1thruVisitorNV()


    separatedFields = [ DummyFieldAccess(a, field.datatype, field.multiplicity)
                         for field in loadStoreFields for a in field.linearAddresses]

    gridIteration(separatedFields, halfWarp, outerSize, visitor)

    return visitor.cycles / outerSize[0] / outerSize[1] / outerSize[2] * 2

def getL1AllocatedLoadBlockVolume(block, grid, loadAddresses):
    visitor = CL128Visitor()
    gridIteration(loadAddresses, block, grid, visitor)
    return visitor.CLs * 128 / grid[0] / grid[1] / grid[2]

def getL1WarpLoadVolume(block, loadAddresses):
    warp = getWarp(32, block)
    grid = tuple(block[i] // warp[i] for i in range(0,3))
    visitor = CL32Visitor()
    gridIteration(loadAddresses, warp, grid, visitor)
    return visitor.CLs * 32



def getL2LoadBlockVolume(block, grid, loadAddresses, fetchSize):
    if fetchSize == 32:
        visitor = CL32Visitor()
    elif fetchSize == 64:
        visitor = CL64Visitor()


    gridIteration(loadAddresses, block, grid, visitor)
    return visitor.CLs * fetchSize / grid[0] / grid[1] / grid[2]

def getL2StoreBlockVolume(block, grid, storeFields):
    warp = getWarp(32, block)

    outerSize = tuple(grid[i] * block[i] // warp[i] for i in range(0, 3))

    separatedFields = [ DummyFieldAccess(a, field.datatype, field.multiplicity)
                         for field in storeFields for a in field.linearAddresses]

    visitor = CL32Visitor()
    gridIteration(separatedFields, warp, outerSize, visitor)
    return visitor.CLs * 32 / grid[0] / grid[1] / grid[2]




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