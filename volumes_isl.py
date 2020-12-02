#!/usr/bin/env python3

import islpy as isl
import random
import timeit


def getMemLoadBlockVolumeISL(
    block, concurrentGrid, totalGrid, loadExprs, validDomain=None
):

    if not validDomain is None:
        validDomainSet = isl.BasicSet(
            "{{[x,y,z] : {0} <= x < {3} and {1} <= y < {4} and {2} <= z < {5} }}".format(
                *validDomain
            )
        )
        # print("{{[x,y,z] : {0} <= x < {3} and {1} <= y < {4} and {2} <= z < {5} }}".format(
        #        *validDomain))

    else:
        validDomainSet = isl.BasicSet("{[x, y, z]}")

    # print(validDomainSet)

    metaGrid = tuple((totalGrid[i] - 1) // concurrentGrid[i] + 1 for i in range(3))

    metaGridPosition = (
        metaGrid[0] // 2,
        metaGrid[1] // 2,
        metaGrid[2] // 2,
    )

    metaGridId = (
        metaGridPosition[0]
        + metaGridPosition[1] * metaGrid[0]
        + metaGridPosition[2] * metaGrid[0] * metaGrid[1]
    )

    prevMetaGridPosition = (
        (metaGridId - 1) % metaGrid[0],
        (metaGridId - 1) // metaGrid[0] % metaGrid[1],
        (metaGridId - 1) // (metaGrid[0] * metaGrid[1]),
    )

    print(metaGrid)
    print(metaGridId)
    print(metaGridPosition)
    print(prevMetaGridPosition)

    concurrentThreads = tuple([block[i] * concurrentGrid[i] for i in range(3)])

    # previousDomainString = " {{[tidx, tidy, tidz] : {0} <= tidx < {3} and {1} <= tidy < {4} and {2} <= tidz < {5}}}".format(
    #    *(metaGridPositionPrev[i] * concurrentThreads[i] for i in range(3)),
    #    *((metaGridPositionPrev[i] + 1) * concurrentThreads[i] for i in range(3)),
    # )

    # print(previousDomainString)
    # previousDomain = isl.BasicSet(previousDomainString).intersect(validDomainSet)
    # print(previousDomain)

    cellCount = 0
    Vcomplete = 0
    Vnew = 0
    Voverlap = 0
    for field in loadExprs:
        accesses = None
        for access in loadExprs[field]:
            mapstring = (
                "{{[tidx, tidy, tidz] -> {}[idx] : idx = floor(({})/4)}}".format(
                    str(field), str(access)
                )
            )
            accessMap = isl.BasicMap(mapstring)

            if not accesses is None:
                accesses = accesses.union(accessMap)
            else:
                accesses = accessMap

        addresses = None
        prevAddresses = None
        cellCount = 0
        for tidz in range(0, block[2]):
            tidzSliceString = " {{[tidx, tidy, tidz] : {0} <= tidx < {3} and {1} <= tidy < {4} and {2} <= tidz < {5}}}".format(
                *(metaGridPosition[i] * concurrentThreads[i] for i in range(2)),
                metaGridPosition[2] * concurrentThreads[2] + tidz,
                *((metaGridPosition[i] + 1) * concurrentThreads[i] for i in range(2)),
                metaGridPosition[2] * concurrentThreads[2] + tidz + 1,
            )

            prevTidzSliceString = " {{[tidx, tidy, tidz] : {0} <= tidx < {3} and {1} <= tidy < {4} and {2} <= tidz < {5}}}".format(
                *(prevMetaGridPosition[i] * concurrentThreads[i] for i in range(2)),
                prevMetaGridPosition[2] * concurrentThreads[2] + tidz,
                *(
                    (prevMetaGridPosition[i] + 1) * concurrentThreads[i]
                    for i in range(2)
                ),
                prevMetaGridPosition[2] * concurrentThreads[2] + tidz + 1,
            )

            tidzSliceDomain = isl.BasicSet(tidzSliceString).intersect(validDomainSet)
            prevTidzSliceDomain = isl.BasicSet(prevTidzSliceString).intersect(
                validDomainSet
            )

            if not addresses is None:
                addresses = addresses.union(tidzSliceDomain.apply(accesses))
            else:
                addresses = tidzSliceDomain.apply(accesses)

            if not prevAddresses is None:
                prevAddresses = prevAddresses.union(prevTidzSliceDomain.apply(accesses))
            else:
                prevAddresses = prevTidzSliceDomain.apply(accesses)
            cellCount += tidzSliceDomain.count_val().to_python()

        Vcomplete += addresses.count_val().to_python()
        Vnew += addresses.subtract(prevAddresses).count_val().to_python()
        Voverlap += addresses.intersect(prevAddresses).count_val().to_python()

    cellCount = max(1, cellCount)  # avoid division by zero
    return (
        Vcomplete * 32 / cellCount * (block[0] * block[1] * block[2]),
        Vnew * 32 / cellCount * (block[0] * block[1] * block[2]),
        Voverlap * 32 / cellCount * (block[0] * block[1] * block[2]),
        cellCount,
    )
    # return Vmem * 32 / cellCount * (block[0] * block[1] * block[2])
