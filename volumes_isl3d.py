#!/usr/bin/env python3

import islpy as isl
import time


def getMemBlockVolumeISL3D(loadExprs, device, block, grid, validDomain, waveBlockCount):

    t1 = time.process_time()
    accessMaps = {}
    for field in loadExprs:

        accessMaps[field] = None
        for access in loadExprs[field]:
            mapstring = "{{[tidx, tidy, tidz] -> {0}[ax, ay, az] :   ax = floor(({1})/4) and ay = {2} and az = {3} }}".format(
                str(field), *[str(a) for a in access]
            )
            if accessMaps[field] is None:
                accessMaps[field] = isl.BasicMap(mapstring)
            else:
                accessMaps[field] = accessMaps[field].union(isl.BasicMap(mapstring))

        accessMaps[field].coalesce()

    if not validDomain is None:
        validDomainSet = isl.BasicSet(
            "{{[x,y,z] : {0} <= x < {3} and {1} <= y < {4} and {2} <= z < {5} }}".format(
                *validDomain
            )
        )
    else:
        validDomainSet = isl.BasicSet("{[x, y, z]}")

    gridBlockCount = grid[0] * grid[1] * grid[2]
    waveBlockCount = waveBlockCount

    def getThreadSet(fromBlock, toBlock):
        currWaveBlockSet = isl.BasicSet(
            "{{[blockId] : {0} <= blockId < {1} and blockId < {2}}}".format(
                fromBlock, toBlock, gridBlockCount
            )
        )

        blockIdToIdxStr = (
            "{{[blockId] -> [blockIdx, blockIdy, blockIdz] : blockIdx = blockId mod {0}"
            " and blockIdy = floor(blockId / {0}) mod {1} and "
            " blockIdz = floor(blockId / {2}) }}".format(
                grid[0], grid[1], grid[1] * grid[0]
            )
        )
        blockIdToIdxMap = isl.BasicMap(blockIdToIdxStr)

        blockToThreadIdxStr = (
            "{{[blockIdx, blockIdy, blockIdz] -> [tidx, tidy, tidz] : "
            "blockIdx * {0} <= tidx < (blockIdx+1) * {0} and "
            "blockIdy * {1} <= tidy < (blockIdy+1) * {1} and "
            "blockIdz * {2} <= tidz < (blockIdz+1) * {2} "
            "}}".format(*block)
        )
        blockToThreadIdxMap = isl.BasicMap(blockToThreadIdxStr)

        currWaveThreadSet = currWaveBlockSet.apply(blockIdToIdxMap).apply(
            blockToThreadIdxMap
        )
        tidDomain = currWaveThreadSet.intersect(validDomainSet)
        return tidDomain

    currThreadSet = getThreadSet(
        gridBlockCount // 2, gridBlockCount // 2 + waveBlockCount
    )
    cellCount = currThreadSet.count_val().to_python()
    Vnew = 0
    currAddressSets = {}
    for field in accessMaps:
        currAddressSets[field] = currThreadSet.apply(accessMaps[field])
        Vnew += currAddressSets[field].count_val().to_python()

    def getVolumes(currAddressSets, prevThreadSet):
        Vold = 0
        Voverlap = 0

        for field in accessMaps:

            prevAddresses = None
            prevAddresses = prevThreadSet.apply(accessMaps[field])

            Vold += prevAddresses.count_val().to_python()
            Voverlap += (
                currAddressSets[field].intersect(prevAddresses).count_val().to_python()
            )

        return Vold, Voverlap

    Vold = 0
    Voverlap = 0
    i = 2
    newPrevLength = 1
    oldPrevLength = 0


    while i < 5 and (Vnew + Vold - Voverlap) * 32 < device.sizeL2:

        t2 = time.process_time()
        newPrevLength = int( newPrevLength * device.sizeL2 / ((Vnew + Vold - Voverlap) * 32) * 1.1 + 0.5)
        if newPrevLength == oldPrevLength:
            break;
        prevThreadSet = getThreadSet(
            gridBlockCount // 2 + waveBlockCount,
            gridBlockCount // 2 + newPrevLength * waveBlockCount,
        )
        Vold, Voverlap = getVolumes(currAddressSets, prevThreadSet)

        print("{:4.1f}  {:4.1f},   {:4.1f},   {:4.1f},   {:5.1f}   {:5.1f}  {:6.1f} ms".format(
            newPrevLength,
            Vnew * 32 / cellCount,
            (Vnew - Voverlap) * 32 / cellCount,
            (Voverlap) * 32 / cellCount,
            Vold * 32 / 1024 / 1024,
            (Vnew + Vold - Voverlap) * 32 / 1024 / 1024,
            (t2-t1)*1000
        ))
        oldPrevLength = newPrevLength
        t1 = t2
        i += 1
    print()

    return Vnew*32, Vold*32, Voverlap*32, cellCount
