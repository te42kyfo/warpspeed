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
        self.CLs += len(set([a // 4 for a in addresses]))


class L1thruVisitor:
    def __init__(self):
        self.cycles = 0

    def count(self, addresses):
        banks = [0] * 16
        maxCycles = 0
        for a in addresses:
            banks[a % 16] += 1
            maxCycles = max(maxCycles, banks[a % 16])
        self.cycles += maxCycles


def gridIteration(
    fields,
    block,
    grid,
    halfWarpVisitor=noneVisitor(),
    tbVisitor=noneVisitor(),
    warpVisitor=noneVisitor(),
    gridVisitor=noneVisitor(),
):

    for field in fields:
        gridAddresses = set()

        for blockId in np.ndindex(grid):
            blockAddresses = set()
            for addressLambda in fields[field]:
                for warpId in range(0, block[0] * block[1] * block[2], 32):
                    warpAddresses = set()
                    for halfWarp in range(0, 32, 16):
                        warpLanes = warpId + halfWarp + np.arange(0, 16)
                        warpX = warpLanes % block[0]
                        warpY = warpLanes // block[0] % block[1]
                        warpZ = warpLanes // block[0] // block[1]
                        halfWarpAddresses = addressLambda(
                            warpX, warpY, warpZ, *blockId, *block
                        )
                        halfWarpVisitor.count(halfWarpAddresses)
                        warpAddresses.update(halfWarpAddresses)
                    warpVisitor.count(warpAddresses)
                    blockAddresses.update(warpAddresses)
            tbVisitor.count(blockAddresses)
            gridAddresses.update(blockAddresses)
        gridVisitor.count(gridAddresses)
