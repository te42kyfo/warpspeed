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
                field.name, *[str(a) for a in access], 32 // field.datatype
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

    def getVolumes(fields):
        VNew = 0
        VOldX = VOldY = VOldZ = 0
        VOverlapX = VOverlapY = VOverlapZ = 0

        for field in fields:
            addSet = currThreadSet.apply(field.accessMap)
            VNew += countSet(addSet) * 32 * field.multiplicity

            xAddSet = xplaneThreadSet.apply(field.accessMap)
            VOldX += countSet(xAddSet) * 32 * field.multiplicity
            VOverlapX += countSet(addSet.intersect(xAddSet)) * 32 * field.multiplicity
            addSet = addSet.subtract(xAddSet)

            yAddSet = yplaneThreadSet.apply(field.accessMap)
            VOldY += countSet(yAddSet) * 32 * field.multiplicity
            VOverlapY += countSet(addSet.intersect(yAddSet)) * 32 * field.multiplicity
            addSet = addSet.subtract(yAddSet)

            zAddSet = zplaneThreadSet.apply(field.accessMap)
            VOldZ += countSet(zAddSet) * 32 * field.multiplicity
            VOverlapZ += countSet(addSet.intersect(zAddSet)) * 32 * field.multiplicity
        return VNew, VOldX, VOverlapX, VOldY, VOverlapY, VOldZ, VOverlapZ

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
    # print(
    #    "{:4.1f},   {:4.1f}   {:4.1f}   {:4.1f},   {:5.1f} {:5.1f} {:5.1f},  {:6.1f} ms".format(
    #        VLoadNew / cellCount,
    #        VLoadOverlapX / cellCount,
    #        VLoadOverlapY / cellCount,
    #        VLoadOverlapZ / cellCount,
    #        (VLoadOldX + VStoreOldX) / 1024 / 1024,
    #        (VLoadOldY + VStoreOldY) / 1024 / 1024,
    #        (VLoadOldZ + VStoreOldZ) / 1024 / 1024,
    #        (t2 - t1) * 1000,
    #    )
    # )

    concurrentThreadCount = (
        blockSize[0]
        * blockSize[1]
        * blockSize[2]
        * waveDim[0]
        * waveDim[1]
        * waveDim[2]
    )

    def rover(Vnew, Vold):
        if Vold == 0:
            coverage = 100
        else:
            coverage = (device.sizeL2 - Vnew) / Vold
        if coverage >= 1.0:
            return 1.0

        if coverage <= 0:
            return 0.0

        return 1.0 * exp(-1000 * exp(-16 * (coverage - 0.3)))

    VnewFactor = (
        waveBlockCount * (blockSize[0] * blockSize[1] * blockSize[2]) / cellCount
    )

    # print(
    #    "Vnew : {:5.3f} -> {:5.3f},  {:5.3f} -> {:5.3f}".format(
    #        VLoadNew / 1024 / 1024, VLoadNew * VnewFactor / 1024 / 1024,
    #        VStoreNew / 1024 / 1024, VStoreNew * VnewFactor / 1024 / 1024
    #    )
    # )

    roverX = rover(0, VLoadOldX + VStoreOldX)
    roverY = rover(0, VLoadOldY + VStoreOldY)
    roverZ = rover(0, VLoadOldZ + VStoreOldZ)

    # print( "Coverages : {:5.3f} {:5.3f} {:5.3f}".format(roverX,roverY,roverZ))

    return (
        VLoadNew,
        VStoreNew,
        (VLoadOldY + VStoreOldY + VLoadOldX + VStoreOldX, VLoadOldZ + VStoreOldZ),
        (VLoadOverlapX + VLoadOverlapY, VLoadOverlapZ),
        (VStoreOverlapX + VStoreOverlapY, VStoreOverlapZ),
        cellCount,
    )
