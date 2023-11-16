#!/usr/bin/env python3

import islpy as isl
import time
from math import ceil, exp


def getMemBlockVolumeISL3D(
    loadFields, storeFields, device, blockSize, grid, validDomain, waveBlockCount
):
    t1 = time.process_time()
    for field in loadFields + storeFields:
        if not getattr(field, "accessMap", None) is None:
            continue
        field.accessMap = None
        for access in field.NDAddresses:
            mapstring = "{{[tidx, tidy, tidz] -> {0}[ax, ay, az] : ax = floor(({1}) / {4}) and ay = {2} and az = {3} }}".format(
                field.name,
                *[str(a) for a in access],
                device.CLFetchSize // field.datatype
            )
            if field.accessMap is None:
                field.accessMap = isl.BasicMap(mapstring)
            else:
                field.accessMap = field.accessMap.union(isl.BasicMap(mapstring))

        field.accessMap.coalesce()

    if validDomain is None:
        validDomain = [0, 0, 0] + [grid[i] * blockSize[i] for i in range(3)]

    validDomainSet = isl.BasicSet(
        "{{[x,y,z] : {0} <= x < {3} and {1} <= y < {4} and {2} <= z < {5} }}".format(
            *validDomain
        )
    )

    gridBlockCount = grid[0] * grid[1] * grid[2]

    waveDim = (
        min(grid[0], waveBlockCount),
        min(
            grid[1],
            ceil(waveBlockCount / ((validDomain[3] - validDomain[0]) / blockSize[0])),
        ),
        min(
            grid[2],
            ceil(
                waveBlockCount
                / (
                    (validDomain[3] - validDomain[0])
                    * (validDomain[4] - validDomain[1])
                    / blockSize[0]
                    / blockSize[1]
                )
            ),
        ),
    )

    def getThreadSet(start, end):
        start = [start[i] * blockSize[i] for i in range(3)]
        end = [end[i] * blockSize[i] for i in range(3)]

        threadSet = isl.BasicSet(
            "{{[threadIdx, threadIdy, threadIdz] : "
            " {0} <= threadIdx < {3} and "
            " {1} <= threadIdy < {4} and "
            " {2} <= threadIdz < {5} }}".format(*start, *end)
        )
        return threadSet.intersect(validDomainSet)

    shiftY = 0
    if (
        grid[1] > 1
        and (validDomain[4] - (grid[1] - waveDim[1]) * blockSize[1]) / blockSize[1]
        < 0.1
    ):
        print(
            "shiftY: "
            + str(
                (validDomain[4] - (grid[1] - waveDim[1]) * blockSize[1]) / blockSize[1]
            )
        )
        print(validDomain[1], grid[1], waveDim[1], blockSize[1])
        shiftY = 1

    currThreadSet = getThreadSet(
        (grid[0] - waveDim[0], grid[1] - waveDim[1] - shiftY, grid[2] // 2),
        (grid[0], grid[1] - shiftY, grid[2] // 2 + waveDim[2]),
    )
    xplaneThreadSet = getThreadSet(
        (grid[0] - 2 * waveDim[0], grid[1] - waveDim[1] - shiftY, grid[2] // 2),
        (grid[0] - waveDim[0], grid[1] - shiftY, grid[2] // 2 + waveDim[2]),
    )
    yplaneThreadSet = getThreadSet(
        (0, max(0, grid[1] - 2 * waveDim[1] - shiftY), grid[2] // 2),
        (grid[0], grid[1] - waveDim[1] - shiftY, grid[2] // 2 + waveDim[2]),
    )
    zplaneThreadSet = getThreadSet(
        (0, 0, grid[2] // 2 - 1), (grid[0], grid[1], grid[2] // 2)
    )

    print("current thread set: \n" + str(currThreadSet))
    print("X,Y,Z plane thread sets:")
    print(xplaneThreadSet)
    print(yplaneThreadSet)
    print(zplaneThreadSet)
    print()

    def countSet(threadSet):
        return threadSet.count_val().to_python()

    cellCount = countSet(currThreadSet)

    fieldVolumes = {}

    def getVolumes(fields):
        VNew = 0
        VOldX = VOldY = VOldZ = 0
        VOverlapX = VOverlapY = VOverlapZ = 0

        # the quadratic domain is larger than the actual wave size by this factor
        qWaveFactor = cellCount / (
            waveBlockCount * blockSize[0] * blockSize[1] * blockSize[2]
        )

        for field in fields:
            addSet = currThreadSet.apply(field.accessMap)
            fieldVNew = countSet(addSet) * device.CLFetchSize * field.multiplicity
            VNew += fieldVNew

            xAddSet = xplaneThreadSet.apply(field.accessMap)
            VOldX += countSet(xAddSet) * device.CLFetchSize * field.multiplicity
            fieldVOverlapX = (
                countSet(addSet.intersect(xAddSet))
                * device.CLFetchSize
                * field.multiplicity
            )
            VOverlapX += fieldVOverlapX
            addSet = addSet.subtract(xAddSet)

            yAddSet = yplaneThreadSet.apply(field.accessMap)
            VOldY += countSet(yAddSet) * device.CLFetchSize * field.multiplicity
            fieldVOverlapY = (
                countSet(addSet.intersect(yAddSet))
                * device.CLFetchSize
                * field.multiplicity
            )
            VOverlapY += fieldVOverlapY
            addSet = addSet.subtract(yAddSet)

            zAddSet = zplaneThreadSet.apply(field.accessMap)
            VOldZ += countSet(zAddSet) * device.CLFetchSize * field.multiplicity
            fieldVOverlapZ = (
                countSet(addSet.intersect(zAddSet))
                * device.CLFetchSize
                * field.multiplicity
            )
            VOverlapZ += fieldVOverlapZ

            fieldVolumes[field.name] = {
                "VNew": fieldVNew / qWaveFactor,
                "VOverlapY": (fieldVOverlapY) / qWaveFactor,
                "VOverlapZ": fieldVOverlapZ / qWaveFactor,
            }

        return (
            VNew / qWaveFactor,
            VOldX / qWaveFactor,
            VOverlapX / qWaveFactor,
            VOldY / qWaveFactor,
            VOverlapY / qWaveFactor,
            VOldZ / qWaveFactor,
            VOverlapZ / qWaveFactor,
        )

    (
        VLoadNew,
        VLoadOldX,
        VLoadOverlapX,
        VLoadOldY,
        VLoadOverlapY,
        VLoadOldZ,
        VLoadOverlapZ,
    ) = getVolumes(loadFields)
    (
        VStoreNew,
        VStoreOldX,
        VStoreOverlapX,
        VStoreOldY,
        VStoreOverlapY,
        VStoreOldZ,
        VStoreOverlapZ,
    ) = getVolumes(storeFields)

    t2 = time.process_time()

    return (
        VLoadNew,
        VStoreNew,
        (VLoadOldY + VStoreOldY, VLoadOldZ + VStoreOldZ),
        (VLoadOverlapX + VLoadOverlapY, VLoadOverlapZ),
        (VStoreOverlapX + VStoreOverlapY, VStoreOverlapZ),
        waveBlockCount * blockSize[0] * blockSize[1] * blockSize[2],
        fieldVolumes,
    )
