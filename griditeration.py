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

    def count(self, addresses):
        self.CLs += np.unique(addresses // 4).size


class CL64Visitor:
    def __init__(self):
        self.CLs = 0

    def count(self, addresses):
        self.CLs += np.unique(addresses // 8).size


class CL128Visitor:
    def __init__(self):
        self.CLs = 0

    def count(self, addresses):
        self.CLs += np.unique(addresses // 16).size

class L1thruVisitor:
    def __init__(self):
        self.cycles = 0

    def count(self, laneAddresses):
        addresses = list(set(laneAddresses))
        banks = [0] * 16
        maxCycles = 0
        for a in addresses:
            bank = int(a) % 16
            banks[bank] += 1
            maxCycles = max(maxCycles, banks[bank])
        self.cycles += max(maxCycles, 0.5 * np.unique(laneAddresses // 16).size)


def gridIteration(fields, innerSize, outerSize, visitor):

    idx = np.arange(32, innerSize[0]+32, dtype=np.int32)
    idy = np.arange(32, innerSize[1]+32, dtype=np.int32)
    idz = np.arange(32, innerSize[2]+32, dtype=np.int32)
    x, y, z = np.meshgrid(idx, idy, idz)

    for field in fields:
        for outerId in np.ndindex(outerSize):
            addresses = np.empty(0)
            for addressLambda in fields[field]:
                addresses = np.concatenate(
                    (addresses, addressLambda(x, y, z, *outerId, *innerSize).ravel())
                )
            visitor.count(addresses)


def getL2LoadBlockVolume(block, grid, loadAddresses):
    visitor = CL32Visitor()
    gridIteration(loadAddresses, block, grid, visitor)
    return visitor.CLs * 32 / grid[0] / grid[1] / grid[2]

def getL1AllocatedLoadBlockVolume(block, grid, loadAddresses):
    visitor = CL128Visitor()
    gridIteration(loadAddresses, block, grid, visitor)
    return visitor.CLs * 128 / grid[0] / grid[1] / grid[2]

def getL1WarpLoadVolume(block, loadAddresses):
    warp = getWarp(32, block)
    grid = tuple(block[i] // warp[i] for i in range(0,3))
    visitor = CL32Visitor()
    gridIteration(loadAddresses, warp, grid, visitor)
    print(warp)
    print (visitor.CLs)
    return visitor.CLs * 32 / block[0] / block[1] / block[2]


def getWarp(warpSize, block):
    warp = (
        min(warpSize, block[0]),
        min(block[1], max(1, warpSize // block[0])),
        min(block[2], max(1, warpSize // block[0] // block[1])),
    )
    return warp


def getL2StoreBlockVolume(block, grid, storeAddresses):
    warp = getWarp(32, block)

    outerSize = tuple(grid[i] * block[i] // warp[i] for i in range(0, 3))

    seperatedFieldAccesses = dict()
    for field in storeAddresses:
        for address in storeAddresses[field]:
            seperatedFieldAccesses[address] = [address]

    visitor = CL32Visitor()
    gridIteration(seperatedFieldAccesses, warp, outerSize, visitor)
    return visitor.CLs * 32 / grid[0] / grid[1] / grid[2]


def getL1Cycles(block, grid, loadStoreAddresses):

    halfWarp = getWarp(16, block)
    outerSize = tuple(grid[i] * block[i] // halfWarp[i] for i in range(0, 3))
    visitor = L1thruVisitor()

    seperatedFieldAccesses = dict()
    for field in loadStoreAddresses:
        for address in loadStoreAddresses[field]:
            seperatedFieldAccesses[address] = [address]

    gridIteration(seperatedFieldAccesses, halfWarp, outerSize, visitor)

    return visitor.cycles / outerSize[0] / outerSize[1] / outerSize[2] * 2


def getMemLoadBlockVolume(block, grid, loadAddresses):
    visitor = CL32Visitor()

    innerSize = tuple(block[i] * grid[i] for i in range(3))
    gridIteration(loadAddresses, innerSize, (1, 1, 1), visitor)
    return visitor.CLs * 32 / grid[0] / grid[1] / grid[2]


def getMemStoreBlockVolume(block, grid, addresses):
    visitor = CL32Visitor()

    innerSize = tuple(block[i] * grid[i] for i in range(3))

    gridIteration(addresses, innerSize, (1, 1, 1), visitor)
    return visitor.CLs * 32 / grid[0] / grid[1] / grid[2]


def getVolumes(kernel, block, grid, validDomain, blockingFactors=(1,1,1)):

    smCount = 80
    sizeL2 = 6 * 1024 * 1024


    blocksPerSM = predict.getBlocksPerSM(block, kernel.registers)
    concurrentGrid = predict.getConcurrentGrid(
        blocksPerSM * smCount, grid
    )

    print("Blocks per SM: " + str(blocksPerSM))
    print("Concurrent Grid: " + str(concurrentGrid))

    truncatedConcurrentGrid = tuple(min(4, c) for c in concurrentGrid)
    threadsPerBlock = block[0] * block[1] * block[2]
    lupsPerBlock = threadsPerBlock * reduce(mul, blockingFactors)
    lupsPerSM = blocksPerSM * lupsPerBlock

    print("Lups/Block: " + str(lupsPerSM))

    results = {}

    results["L1AllocatedLoad"] = getL1AllocatedLoadBlockVolume(block, truncatedConcurrentGrid, kernel.genLoads()) * blocksPerSM
    results["L1Load"] = getL2StoreBlockVolume(block, truncatedConcurrentGrid, kernel.genLoads()) / lupsPerBlock
    results["L1WarpLoad"] = getL1WarpLoadVolume(block, kernel.genLoads()) / reduce(mul, blockingFactors)

    results["L2Load"] = (
        getL2LoadBlockVolume(block, truncatedConcurrentGrid, kernel.genLoads())
        / lupsPerBlock
    )

    results["L2Store"] = (
        getL2StoreBlockVolume(block, truncatedConcurrentGrid, kernel.genStores())
        / lupsPerBlock
    )

    results["memLoad"] = (
        getMemLoadBlockVolume(block, concurrentGrid, kernel.genLoads())
        / lupsPerBlock
    )

    results["memStore"] = (
        getMemStoreBlockVolume(block, concurrentGrid, kernel.genStores())
        / lupsPerBlock
    )

    vMemNew, vMemOld, vMemOverlap, cellCount = getMemLoadBlockVolumeISL(
        block,
        concurrentGrid,
        grid,
        kernel.genLoadExprs(),
        validDomain,
    )

    concurrentBlocks = predict.getBlocksPerSM(block, kernel.registers) * smCount
    vMemNew *= concurrentBlocks
    vMemOld *= concurrentBlocks
    vMemOverlap *= concurrentBlocks

    vMemStore = results["memStore"] * concurrentBlocks * lupsPerBlock
    vL2Store = results["L2Store"] * concurrentBlocks * lupsPerBlock
    vL2Load = results["L2Load"] * concurrentBlocks * lupsPerBlock

    vL2LoadSM = results["L2Load"] * lupsPerSM

    print("Allocated L1 Volume / SM: {:.0f} kB".format((results["L1AllocatedLoad"] ) / 1024))
    print("L1 Load Volume: {:.0f} B / Lup".format((results["L1Load"] ) ))
    print("L1 Warp Load Volume: {:.0f} B / Lup".format((results["L1WarpLoad"] ) ))

    vL1CapacityMiss = max(0, (1 - (128*1024) / ( results["L1AllocatedLoad"] ))) * (results["L1Load"] - results["L2Load"]) * 0.25
    print("L1 capacity miss: {:.1f} B/Lup".format(vL1CapacityMiss))
    results["L2LoadExt"] = results["L2Load"] + vL1CapacityMiss

    vMem = 0

    vMemTotal = vMemNew + vMemOld - vMemOverlap

    effectiveL2 = max(sizeL2 * 0.5, sizeL2 - vMemStore)
    if vMemTotal < effectiveL2:
        vMem = vMemNew - vMemOverlap
    elif vMemNew < effectiveL2:
        vMem = vMemNew - vMemOverlap * (1 - (vMemTotal - effectiveL2) / vMemOld)
    else:
        vMem = vMemNew



    results["L2LoadAllocated"] = vMem + vMemStore
    results["memLoadISL"] = vMem / concurrentBlocks / lupsPerBlock

    vMemCapacityMiss = min(1, max(0, (1 - effectiveL2 / ( vMem )))) * (results["L2LoadExt"] - results["memLoadISL"])
    results["memLoadISLext"] = (vMem ) / concurrentBlocks / lupsPerBlock + vMemCapacityMiss

    vStoreEvicted = (vL2Store - vMemStore) * 0.2

    results["memStoreExt"] = (
        (vMemStore + vStoreEvicted) / concurrentBlocks / lupsPerBlock
    )

    print(
        "Load Data Volumes (new, old, overlap): {:.0f}  {:.0f}  {:.0f} kb".format(
            vMemNew / 1024,
            vMemOld / 1024,
            vMemOverlap / 1024,
        )
    )

    print(
        "Store Data Volumes (mem, L2, evicted): {:.0f} {:.0f} {:.0f} kb  ".format(
            vMemStore / 1024, vL2Store / 1024, vStoreEvicted / 1024
        )
    )

    results["L1cycles"] = getL1Cycles(
        block, truncatedConcurrentGrid, {**kernel.genLoads(), **kernel.genStores()}
    )
    return results
