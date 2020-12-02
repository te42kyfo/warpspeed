#!/usr/bin/env python3

import numpy as np

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

    idx = np.arange(32, innerSize[0] + 32, dtype=np.int32)
    idy = np.arange(32, innerSize[1] + 32, dtype=np.int32)
    idz = np.arange(32, innerSize[2] + 32, dtype=np.int32)
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


def getVolumes(kernel, block, grid, validDomain):

    concurrentGrid = predict.getConcurrentGrid(
        predict.getBlocksPerSM(block, 32) * 80, grid
    )
    truncatedConcurrentGrid = tuple(min(4, c) for c in concurrentGrid)
    threadsPerBlock = block[0] * block[1] * block[2]

    results = {}

    results["L2Load"] = (
        getL2LoadBlockVolume(block, truncatedConcurrentGrid, kernel.genLoads())
        / threadsPerBlock
    )

    results["L2Store"] = (
        getL2StoreBlockVolume(block, truncatedConcurrentGrid, kernel.genStores())
        / threadsPerBlock
    )

    results["memLoad"] = (
        getMemLoadBlockVolume(block, concurrentGrid, kernel.genLoads())
        / threadsPerBlock
    )

    results["memStore"] = (
        getMemStoreBlockVolume(block, concurrentGrid, kernel.genStores())
        / threadsPerBlock
    )

    vMemComplete, vMemNew, vMemOverlap, cellCount = getMemLoadBlockVolumeISL(
        block,
        concurrentGrid,
        grid,
        kernel.genLoadExprs(),
        validDomain,
    )

    concurrentBlocks = predict.getBlocksPerSM(block, 32) * 80
    vMemComplete *= concurrentBlocks
    vMemNew *= concurrentBlocks
    vMemOverlap *= concurrentBlocks

    sizeL2 = 6 * 1024 * 1024

    vMemStore = results["memStore"] * concurrentBlocks * threadsPerBlock
    vL2Store = results["L2Store"] * concurrentBlocks * threadsPerBlock
    vL2Load = results["L2Load"] * concurrentBlocks * threadsPerBlock

    vMem = 0

    if vMemNew < sizeL2:
        vMem = vMemNew
    elif vMemNew < sizeL2:
        vMem = vMemNew + vMemOverlap * ((2 * vMemNew - sizeL2) / vMemNew) ** 2 * 0.5
    else:
        vMem = vMemNew + vMemOverlap * (1 - 0.5 * (sizeL2 / vMemNew) ** 2)

    vL2 = results["L2Load"] * concurrentBlocks * threadsPerBlock

    vMemEvicted = 0
    if vMemComplete + vMemStore > sizeL2 and vMemComplete > 0:
        vMemEvicted = (
            (vL2 - vMemComplete)
            * (vMemComplete - sizeL2 * (1 - vMemStore / vMemComplete))
            / vMemComplete
        )

    # vMem += vMemEvicted

    results["memLoadISL"] = vMem / concurrentBlocks / threadsPerBlock
    results["memLoadISLext"] = (vMem + vMemEvicted) / concurrentBlocks / threadsPerBlock

    vStoreEvicted = (vL2Store - vMemStore) * 0.2

    results["memStoreExt"] = (
        (vMemStore + vStoreEvicted) / concurrentBlocks / threadsPerBlock
    )

    print(
        "Load Data Volumes: {:.0f} {:.0f} {:.0f} kb".format(
            vMemComplete / 1024,
            vL2 / 1024,
            vMemEvicted / 1024,
        )
    )

    print(
        "Store Data Volumes: {:.0f} {:.0f} {:.0f} kb  ".format(
            vMemStore / 1024, vL2Store / 1024, vStoreEvicted / 1024
        )
    )

    results["L1cycles"] = getL1Cycles(
        block, truncatedConcurrentGrid, {**kernel.genLoads(), **kernel.genStores()}
    )
    return results
