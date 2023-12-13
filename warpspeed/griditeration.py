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


class CLVisitor:
    def __init__(self, clSize):
        self.CLs = 0
        self.clSize = clSize

    def count(self, addresses, multiplicity=1, datatype=8):
        self.CLs += np.unique(addresses // self.clSize).size * multiplicity


class CLVisitorNoMultiplicity:
    def __init__(self, clSize):
        self.CLs = 0
        self.clSize = clSize

    def count(self, addresses, multiplicity=1, datatype=8):
        self.CLs += np.unique(addresses // self.clSize).size


class PageVisitor:
    def __init__(self, pageSize):
        self.pageSize = pageSize
        self.pages = 0

    def count(self, addresses, multiplicity=1, datatype=8):
        self.pages += np.unique(addresses // (self.pageSize)).size * multiplicity


class CollectingVisitor:
    def __init__(self, clSize):
        self.clSize = clSize
        self.CLs = set()

    def count(self, addresses, multiplicity=1, datatype=8):
        self.CLs.update(list(addresses // self.clSize))


class OverlapVisitor:
    def __init__(self, collection):
        self.collection = collection
        self.overlapCLs = 0

    def count(self, addresses, multiplicity=1, datatype=8):
        self.overlapCLs += (
            len(
                self.collection.CLs.intersection(
                    set(addresses // self.collection.clSize)
                )
            )
            * multiplicity
        )


def countBankConflicts(totalLength, laneAddresses, datatype=8):
    addresses = list(set([l // datatype for l in laneAddresses]))

    banks = [0] * (totalLength // datatype)
    maxCycles = 0
    for a in addresses:
        bank = int(a) % (totalLength // datatype)
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
            countBankConflicts(128, laneAddresses, datatype)
        ) * multiplicity

        self.tagCycles += (
            (max(1, np.unique(laneAddresses // 128).size) + 2) * multiplicity / 4
        )


class L1thruVisitorCDNA:
    def __init__(self):
        self.dataPipeCycles = 0
        self.tagCycles = 0
        self.uniform = 0
        self.coalesced = 0
        self.cycles = 0

    def count(self, laneAddresses, multiplicity=1, datatype=8):
        addresses = list(set([l // 64 for l in laneAddresses]))
        # print(addresses)
        # print()
        cycles = np.unique(addresses).size / 1.5

        # print("cycles ", cycles)

        if datatype != 4:
            cycles = max(16 / 4, cycles)

        # print("cycles ", cycles)

        self.dataPipeCycles += (
            countBankConflicts(64, laneAddresses, datatype)
        ) * multiplicity

        self.tagCycles += (cycles) * multiplicity


class DummyFieldAccess:
    def __init__(self, address, datatype, multiplicity):
        self.linearAddresses = [address]
        self.datatype = datatype
        self.multiplicity = multiplicity


def gridIteration(fields, innerSize, outerSize, visitor, startBlock=(0, 0, 0)):
    idx = np.arange(0, innerSize[0], dtype=np.int32)
    idy = np.arange(0, innerSize[1], dtype=np.int32)
    idz = np.arange(0, innerSize[2], dtype=np.int32)
    x, y, z = np.meshgrid(idx, idy, idz)

    for field in fields:
        for outerId in np.ndindex(outerSize):
            outerId += np.array(startBlock)
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
            # print(addresses)
            byteAddresses = np.array(
                [a + i * 4 for i in range(field.datatype // 4) for a in addresses]
            )
            # print(byteAddresses)
            visitor.count(byteAddresses, field.multiplicity, field.datatype)


def getWarp(warpSize, block):
    warp = (
        min(warpSize, block[0]),
        min(block[1], max(1, warpSize // block[0])),
        min(block[2], max(1, warpSize // block[0] // block[1])),
    )
    return warp


def getL1Cycles(block, grid, loadStoreFields, device):
    warpSize = 16 if device.L1Model == "CDNA" else device.warpSize

    fieldCycles = {}

    L1Components = namedtuple("L1Components", "dataPipeCycles tagCycles total")

    dataPipeCycles = 0
    tagCycles = 0
    totalCycles = 0
    for field in loadStoreFields:
        separatedFields = [
            DummyFieldAccess(a, d, field.multiplicity)
            for a, d in zip(field.linearAddresses, field.datatypes)
        ]

        warp = getWarp(warpSize, block)

        if device.L1Model == "CDNA":
            visitor = L1thruVisitorCDNA()
        else:
            visitor = L1thruVisitorNV()

        outerSize = [min(64, block[i] * grid[i]) // warp[i] for i in [0, 1, 2]]
        outerSize[0] = max(16 // block[0], outerSize[0])
        outerSize = tuple(outerSize)

        # print("warp ", warp)
        # print("grid ", outerSize)
        gridIteration(separatedFields, warp, outerSize, visitor)

        fieldDataPipeCycles = (
            visitor.dataPipeCycles
            / outerSize[0]
            / outerSize[1]
            / outerSize[2]
            / warpSize
        )
        fieldTagCycles = (
            visitor.tagCycles / outerSize[0] / outerSize[1] / outerSize[2] / warpSize
        )
        fieldCycles[field.name] = L1Components(
            fieldDataPipeCycles,
            fieldTagCycles,
            max(fieldDataPipeCycles, fieldTagCycles),
        )

        dataPipeCycles += fieldDataPipeCycles
        tagCycles += fieldTagCycles
        totalCycles += max(fieldDataPipeCycles, fieldTagCycles)
    # print(visitor.cycles / outerSize[0] / outerSize[1] / outerSize[2])
    # print(visitor.tagCycles / outerSize[0] / outerSize[1] / outerSize[2])
    # print(visitor.dataPipeCycles / outerSize[0] / outerSize[1] / outerSize[2])
    # print(visitor.uniform / outerSize[0] / outerSize[1] / outerSize[2])
    # print(visitor.coalesced / outerSize[0] / outerSize[1] / outerSize[2])

    fieldCycles["total"] = L1Components(dataPipeCycles, tagCycles, totalCycles)
    return fieldCycles


def getL1AllocatedLoadBlockVolume(block, grid, loadAddresses, CLAllocationSize):
    visitor = CLVisitorNoMultiplicity(CLAllocationSize)
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
    fieldVolumes = {}

    totalVolume = 0
    for field in loadAddresses:
        if fetchSize == 32:
            visitor = CL32Visitor()
        elif fetchSize == 64:
            visitor = CL64Visitor()

        gridIteration([field], block, grid, visitor)
        totalVolume += visitor.CLs * fetchSize / grid[0] / grid[1] / grid[2]
        fieldVolumes[field.name] = visitor.CLs * fetchSize / grid[0] / grid[1] / grid[2]
    fieldVolumes["total"] = totalVolume

    return fieldVolumes


def getL2LoadOverlapBlockVolume(block, totalGrid, loadAddresses, fetchSize):
    currentBlock = [g // 2 for g in totalGrid]
    blockId = (
        currentBlock[0]
        + currentBlock[1] * totalGrid[0]
        + currentBlock[2] * totalGrid[1] * totalGrid[0]
    )
    prevBlockId = blockId
    prevBlocks = []
    i = 0
    while prevBlockId > 0 and i < 40:
        prevBlockId = max(0, prevBlockId - 108)
        prevBlocks.append(
            [
                prevBlockId % totalGrid[0],
                prevBlockId // totalGrid[0] % totalGrid[1],
                prevBlockId // (totalGrid[0] * totalGrid[1]),
            ]
        )
        i += 1

    fieldVolumes = {}

    totalOverlap = 0
    for field in loadAddresses:
        collector = CollectingVisitor(fetchSize)
        visitor = OverlapVisitor(collector)

        gridIteration([field], block, (1, 1, 1), collector, currentBlock)
        for prevBlock in prevBlocks:
            gridIteration([field], block, (1, 1, 1), visitor, prevBlock)

        fieldVolumes[field.name] = visitor.overlapCLs * fetchSize / len(prevBlocks)
        totalOverlap += fieldVolumes[field.name]

    fieldVolumes["total"] = totalOverlap
    return fieldVolumes


def getL2StoreBlockVolume(block, grid, storeFields, device):
    warp = getWarp(device.warpSize, block)

    outerSize = tuple(grid[i] * block[i] // warp[i] for i in range(0, 3))

    fieldVolume = {}
    totalVolume = 0
    for field in storeFields:
        separatedFields = [
            DummyFieldAccess(a, d, field.multiplicity)
            for a, d in zip(field.linearAddresses, field.datatypes)
        ]

        visitor = CLVisitor(device.CLFetchSize)
        gridIteration(separatedFields, warp, outerSize, visitor)
        volume = visitor.CLs * device.CLFetchSize / grid[0] / grid[1] / grid[2]
        fieldVolume[field.name] = volume
        totalVolume += volume

    fieldVolume["total"] = totalVolume
    return fieldVolume


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
