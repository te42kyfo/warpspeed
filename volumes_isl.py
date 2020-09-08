#!/usr/bin/env python3

import islpy as isl


def getMemLoadBlockVolumeISL(block, grid, loadExprs):

    totalGrid = tuple(block[i] * grid[i] for i in range(3))

    domainString = " {{[tidx, tidy, tidz] : 0 <= tidx < {} and 0 <= tidy < {} and 0 <= tidz < {}}}".format(
        totalGrid[0], totalGrid[1], totalGrid[2]
    )
    print(domainString)
    domain = isl.BasicSet(domainString)
    accesses = None
    for field in loadExprs:
        for access in loadExprs[field]:
            mapstring = (
                "{{[tidx, tidy, tidz] -> {}[idx] : idx = floor(({}) / 4)}}".format(
                    str(field), str(access)
                )
            )
            accessMap = isl.BasicMap(mapstring)
            if not accesses is None:
                accesses = accesses.union(accessMap)
            else:
                accesses = accessMap
    print(domain.apply(accesses))
    print(domain.apply(accesses).coalesce())
    clcount = domain.apply(accesses).count_val().to_python()
    print(clcount)
    return (
        domain.apply(accesses).count_val().to_python()
        * 32
        / (grid[0] * grid[1] * grid[2])
    )
