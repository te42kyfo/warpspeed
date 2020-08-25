#!/usr/bin/env python3
import numpy as np
import sympy
import scipy.stats as stats
from math import ceil, floor, log, exp


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


def predictPerformanceV1(kernel, block, grid, registers):

    threadsPerBlock = block[0] * block[1] * block[2]
    blocksPerSM = min(32, int(2 ** 16 / (threadsPerBlock * max(registers, 32))))
    warpsPerSM = blocksPerSM * ((threadsPerBlock - 1) // 32 + 1)
    warpsPerBlock = ceil(threadsPerBlock / 32)

    SMcount = 80
    clock = 1.38
    cl_size = 32

    L2CLsVisitor = CL32Visitor()
    L1CLsVisitor = CL32Visitor()

    gridIteration(
        kernel.genAddresses(),
        block,
        grid,
        tbVisitor=L2CLsVisitor,
        warpVisitor=L1CLsVisitor,
    )

    L2CLs = L2CLsVisitor.CLs / grid[0] / grid[1] / grid[2]
    L1CLs = L1CLsVisitor.CLs / grid[0] / grid[1] / grid[2]

    L2CLs += threadsPerBlock / 4
    L1LoadCycles = L1CLs / 4 / warpsPerBlock

    Tint = 40 * 5 * max(1, warpsPerSM / 12)
    TL1thru = L1LoadCycles * warpsPerSM
    TDP = kernel.flops * max(1, (warpsPerSM / 8)) * 8
    Tlat_mem = (
        max(271, (warpsPerSM * 32 * 16 * SMcount) / 780 * clock)
        if kernel.bytes > 0
        else 0
    )
    Tlat_L2 = (
        max(200, (L2CLs * blocksPerSM * SMcount * cl_size) / 2000 * clock)
        if kernel.bytes > 0
        else 0
    )
    Ttotal = Tint + max(TDP, TL1thru) + Tlat_mem + Tlat_L2
    print(" Tint  TL1thru   TDP  Tlat_mem  Tlat_L2  Ttotal")
    print(
        "{:5.0f}{:9.0f}  {:4.0f}  {:8.0f}  {:7.0f}  {:6.0f}".format(
            Tint, TL1thru, TDP, Tlat_mem, Tlat_L2, Ttotal
        )
    )

    delta = 100
    for i in range(0, 200):
        Tint = 40 * 5 * max(1, Tint / Ttotal * warpsPerSM / 12)
        TL1thru = L1LoadCycles * max(1, TL1thru / Ttotal * warpsPerSM)
        TDP = kernel.flops * max(1, warpsPerSM * (TDP / Ttotal) / 8) * 8
        Tlat_mem = (
            max(271, Tlat_mem / Ttotal * (warpsPerSM * 32 * 16 * SMcount) / 780 * clock)
            if kernel.bytes > 0
            else 0
        )
        Tlat_L2 = (
            max(
                200,
                Tlat_L2
                / Ttotal
                * (L2CLs * blocksPerSM * SMcount * cl_size)
                / 2000
                * clock,
            )
            if kernel.bytes > 0
            else 0
        )
        new_Ttotal = Tint + max(TDP, TL1thru) + Tlat_mem + Tlat_L2
        delta = abs(new_Ttotal - Ttotal)
        Ttotal = new_Ttotal

        if i > 100 and delta < 0.01:
            break

    print(
        "{:5.0f}{:9.0f}  {:4.0f}  {:8.0f}  {:7.0f}  {:6.0f}".format(
            Tint, TL1thru, TDP, Tlat_mem, Tlat_L2, Ttotal
        )
    )
    return kernel.flops * blocksPerSM * threadsPerBlock * (clock * SMcount / Ttotal)


def maxOverlap(t, w, p):
    return max(1, t * w / p)


def randomOverlap(t, w, p):

    utilization = 0
    for k in range(0, w):
        prob = stats.binom.pmf(k, w - 1, t)
        utilization += prob * (max(1, (k + 1) / p))
        # print("{} {:4.2f} {:4.2f}".format(k, prob, max(1, (k+1)/p)))
    # print("{} {} {} {:5.3f}".format(t, w, p, utilization))
    return utilization


def L2Latency(wdev, payload, clock):
    # return max(200, (payload * wdev) / 1500 * clock)
    x = wdev * payload / 8 / 32
    # lat = 210 + 102* log(1 + exp((x-1400)*0.00138))
    lat = max(237, 175 + 105 * log(1 + exp((x - 900) * 0.00140)))
    # print("{:6.0f} {:6.0f} {:6.1f}".format(x, lat, payload / 8 / 32))
    return lat


def memLatency(wdev, payload, clock):
    # return max(250, wdev*payload / 780 * clock)
    x = wdev * payload / 8 / 32
    lat = max(210, 0 + 270 * log(1 + exp((x - 100) * 0.0017)))
    # print("{:6.0f} {:6.0f} {:6.1f}".format(x, lat, payload / 8 / 32))
    return lat


def predictPerformance(kernel, block, grid, overlap=randomOverlap):
    threadsPerBlock = block[0] * block[1] * block[2]
    blocksPerSM = min(32, int(2 ** 16 / (threadsPerBlock * max(kernel.registers, 32))))
    warpsPerSM = blocksPerSM * ((threadsPerBlock - 1) // 32 + 1)
    warpsPerBlock = ceil(threadsPerBlock / 32)
    warpsPerQuadrant = ceil(warpsPerSM / 4)

    SMcount = 80
    clock = 1.38
    cl_size = 32
    blockShedRate = 0.46

    L2CLsVisitor = CL32Visitor()
    L1Visitor = L1thruVisitor()

    gridIteration(
        kernel.genAddresses(),
        block,
        grid,
        tbVisitor=L2CLsVisitor,
        halfWarpVisitor=L1Visitor,
    )

    L2CLs = L2CLsVisitor.CLs / grid[0] / grid[1] / grid[2]

    L2CLs += threadsPerBlock / 4

    L1LoadCycles = L1Visitor.cycles / grid[0] / grid[1] / grid[2] / warpsPerBlock

    print(L2CLs / threadsPerBlock * 32)
    print(L1LoadCycles)

    Tint = 40 * 4 * max(1, warpsPerSM / 8)
    TL1thru = L1LoadCycles * blocksPerSM
    TDP = kernel.flops * max(1, (warpsPerQuadrant / 8)) * 8
    Tlat_mem = (
        memLatency(warpsPerSM * SMcount, 16 * 32, clock) if kernel.bytes > 0 else 0
    )
    Tlat_L2 = (
        L2Latency(warpsPerSM * SMcount, L2CLs / warpsPerBlock * cl_size, clock)
        if kernel.bytes > 0
        else 0
    )
    Tblocksched = SMcount / 0.5 * blocksPerSM
    Ttotal = Tblocksched + Tint + max(TDP, TL1thru) + Tlat_mem + Tlat_L2
    print("Tblocksched  Tint TL1thru   TDP Tlat_mem Tlat_L2 Ttotal")
    print(
        "{:11.0f} {:5.0f} {:7.0f} {:5.0f} {:8.0f} {:7.0f} {:6.0f}".format(
            Tblocksched, Tint, TL1thru, TDP, Tlat_mem, Tlat_L2, Ttotal
        )
    )

    delta = 100
    for i in range(0, 200):
        Tint = 40 * 4 * overlap(Tint / Ttotal, warpsPerSM, 8)
        TL1thru = L1LoadCycles * overlap(TL1thru / Ttotal, warpsPerSM, 1)
        TDP = kernel.flops * 8 * overlap(TDP / Ttotal, warpsPerQuadrant, 2)
        Tlat_mem = (
            memLatency(Tlat_mem / Ttotal * warpsPerSM * SMcount, 16 * 32, clock)
            if kernel.bytes > 0
            else 0
        )
        Tlat_L2 = (
            L2Latency(
                Tlat_L2 / Ttotal * warpsPerSM * SMcount,
                L2CLs / warpsPerBlock * cl_size,
                clock,
            )
            if kernel.bytes > 0
            else 0
        )
        new_Ttotal = Tblocksched + Tint + max(TDP, TL1thru) + Tlat_mem + Tlat_L2
        delta = abs(new_Ttotal - Ttotal)
        Ttotal = new_Ttotal

        if i > 100 and delta < 0.01:
            break

    print(
        "{:11.0f} {:5.0f} {:7.0f} {:5.0f} {:8.0f} {:7.0f} {:6.0f}".format(
            Tblocksched, Tint, TL1thru, TDP, Tlat_mem, Tlat_L2, Ttotal
        )
    )
    return blocksPerSM * threadsPerBlock * (clock * SMcount / Ttotal)
