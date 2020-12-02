#!/usr/bin/env python3
import sys

sys.path.append("../pystencils")
sys.path.append("../genpredict")

import cProfile
from functools import partial, reduce
from operator import mul
import pycuda.driver as drv
import measure_metric.measureMetric as measureMetric
import math
from predict import *
from timeit import timeit


def timeKernel(run_func):
    start = drv.Event()
    end = drv.Event()
    start.record()
    run_func()
    end.record()
    end.synchronize()
    return end.time_since(start)


def benchKernel(run_func):
    run_func()
    repeats = [timeKernel(run_func) for i in range(0, 5)]
    repeats.sort()
    time = repeats[len(repeats) // 2]
    return time


def computeMetrics(wsKernel, block, grid, validDomain=None):
    concurrentGrid = getConcurrentGrid(getBlocksPerSM(block, 32) * 80, grid)
    truncatedConcurrentGrid = tuple(min(4, c) for c in concurrentGrid)
    threadsPerBlock = block[0] * block[1] * block[2]

    results = {}

    results["L2Load"] = (
        getL2LoadBlockVolume(block, truncatedConcurrentGrid, wsKernel.genLoads())
        / threadsPerBlock
    )
    # print(
    #    "L2LoadBlockVolume: {:.1f}ms".format(
    #        timeit(
    #            lambda: getL2LoadBlockVolume(
    #                block, truncatedConcurrentGrid, wsKernel.genLoads()
    #            ),
    #            number=1,
    #        )
    #        * 1000
    #    )
    # )

    results["L2Store"] = (
        getL2StoreBlockVolume(block, truncatedConcurrentGrid, wsKernel.genStores())
        / threadsPerBlock
    )
    # print(
    #     "L2StoreBlockVolume: {:.1f}ms".format(
    #         timeit(
    #             lambda: getL2StoreBlockVolume(
    #                 block, truncatedConcurrentGrid, wsKernel.genStores()
    #             ),
    #             number=1,
    #         )
    #         * 1000
    #     )
    # )

    results["memLoad"] = (
        getMemLoadBlockVolume(block, concurrentGrid, wsKernel.genLoads())
        / threadsPerBlock
    )
    # print(
    #     "getMemLoadBLockVolume: {:.1f}ms".format(
    #         timeit(
    #             lambda: getMemLoadBlockVolume(
    #                 block, concurrentGrid, wsKernel.genLoads()
    #             ),
    #             number=1,
    #         )
    #         * 1000
    #     )
    # )

    results["memStore"] = (
        getMemStoreBlockVolume(block, concurrentGrid, wsKernel.genStores())
        / threadsPerBlock
    )
    # print(
    #     "getMemStoreBLockVolume: {:.1f}ms".format(
    #         timeit(
    #             lambda: getMemStoreBlockVolume(
    #                 block, concurrentGrid, wsKernel.genStores()
    #             ),
    #             number=1,
    #         )
    #         * 1000
    #     )
    # )

    results["memLoadISL"] = (
        getMemLoadBlockVolumeISL(
            block,
            concurrentGrid,
            grid,
            wsKernel.genLoadExprs(),
            validDomain,
        )
        / threadsPerBlock
    )
    # print(
    #     "getMemLoadBLockVolumeISL: {:.1f}ms".format(
    #         timeit(
    #             lambda: getMemLoadBlockVolumeISL(
    #                 block,
    #                 concurrentGrid,
    #                 grid,
    #                 wsKernel.genLoadExprs(),
    #                 [0, 0, 0, 500 - 4, 500 - 4, 500 - 4],
    #             ),
    #             number=1,
    #         )
    #         * 1000
    #     )
    # )

    results["L1cycles"] = getL1Cycles(
        block, truncatedConcurrentGrid, {**wsKernel.genLoads(), **wsKernel.genStores()}
    )
    # print(
    #     "getL1Cycles: {:.1f}ms".format(
    #         timeit(
    #             lambda: getL1Cycles(
    #                 block,
    #                 truncatedConcurrentGrid,
    #                 {**wsKernel.genLoads(), **wsKernel.genStores()},
    #             ),
    #             number=1,
    #         )
    #         * 1000
    #     )
    # )
    return results


def measureMetrics(runFunc, size):
    lupCount = reduce(mul, size)

    measurements = []
    for i in range(0, 3):
        measureMetric.measureBandwidthStart()
        runFunc()
        measurements.append(measureMetric.measureMetricStop())

    medians = [
        sorted([m[i] for m in measurements])[len(measurements) // 2] for i in range(4)
    ]

    results = dict()

    results["memLoad"] = medians[0] / lupCount
    results["memStore"] = medians[1] / lupCount
    results["L2Load"] = medians[2] * 32 / lupCount
    results["L2Store"] = medians[3] * 32 / lupCount

    return results
