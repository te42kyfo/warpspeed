#!/usr/bin/env python3

from meas_utils import *
from griditeration import *
from volumes_isl import *
from column_print import *


class DeviceVolta:
    def __init__(self):
        self.smCount = 80
        self.sizeL2 = 6 * 1024 * 1024
        self.sizeL1 = 128*1024


class LaunchConfig:
    def __init__(self, kernel, block, domain, blocking_factors, device):
        self.kernel = kernel
        self.block = block
        self.grid = tuple( (domain[i]-1) // block[i] + 1 for i in range(len(block)))
        self.domain = domain
        self.blocking_factors = blocking_factors
        self.device = device


        self.blocksPerSM = predict.getBlocksPerSM(block, kernel.registers)

        self.waveSize = predict.getConcurrentGrid(
            self.blocksPerSM * device.smCount, self.grid
        )

        self.truncatedWaveSize = tuple(min(4, c) for c in self.waveSize)
        self.threadsPerBlock = block[0] * block[1] * block[2]
        self.lupsPerThread = reduce(mul, blocking_factors)


    def stringKey(self, key, labelWidth, valueWidth):

        if key in self.__dict__:
            return "{:{labelWidth}}: {:{valueWidth}}   ".format(str(key), str(self.__dict__[key]),  labelWidth=labelWidth, valueWidth=valueWidth)
        else:
            return "{:{width}}".format(" ", width=labelWidth + valueWidth)

    def __str__(self):
        columns = [["block", "grid", "waveSize", "truncatedWaveSize"],
                    ["blocking_factors", "threadsPerBlock", "blocksPerSM"]]
        return columnPrint(self, columns)



class BasicMetrics:
    def __init__(self, lc, device, kernel):
        self.L1Cycles = getL1Cycles(lc.block, lc.truncatedWaveSize, {**kernel.genLoads(), **kernel.genStores()})
        self.blockL1LoadAllocation = getL1AllocatedLoadBlockVolume(lc.block, lc.truncatedWaveSize, kernel.genLoads())
        self.blockL1Load = getL2StoreBlockVolume(lc.block, lc.truncatedWaveSize, kernel.genLoads())
        self.warpL1Load = getL1WarpLoadVolume(lc.block, kernel.genLoads())
        self.blockL2Load =  getL2LoadBlockVolume(lc.block, lc.truncatedWaveSize, kernel.genLoads())
        self.blockL2Store =  getL2StoreBlockVolume(lc.block, lc.truncatedWaveSize, kernel.genStores())
        self.waveMemLoad = getMemLoadWaveVolume(lc.block, lc.waveSize, kernel.genLoads())
        self.waveMemStore = getMemStoreWaveVolume(lc.block, lc.waveSize, kernel.genStores())
        self.waveMemLoadISL, self.waveMemLoadOld, self.waveMemOverlap, self.waveValidCells = getMemLoadBlockVolumeISL(lc.block, lc.waveSize, lc.grid, kernel.genLoadExprs(), [0,0,0] + lc.domain)

    def stringKey(self, key, labelWidth, valueWidth):
        kB = key != "L1Cycles" and key != "waveValidCells"
        if key in self.__dict__:
            string = "{:{labelWidth}}: {:{valueWidth}.0f} {:2}   ".format(str(key),  self.__dict__[key] / (1024 if kB else 1), "kB" if kB else "",
                                                                          labelWidth=labelWidth, valueWidth=valueWidth)
        else:
            string = "{:{width}}".format(" ", width=labelWidth+valueWidth)
        return string

    def __str__(self):
        columns = [["blockL1LoadAllocation", "blockL1Load", "warpL1Load", "blockL2Load"],
                   ["waveMemLoad", "waveMemLoadISL", "waveMemLoadOld", "waveMemOverlap"],
                   ["waveValidCells", "L1Cycles", "blockL2Store", "waveMemStore"]]
        return columnPrint(self, columns)

class DerivedMetrics:
    def __init__(self, lc, basic, device):
        self.lc = lc
        self.basic = basic
        self.device = device

    def L1Cycles(self):
        return self.basic.L1Cycles

    def L1Load(self):
        return self.basic.blockL1Load / self.lc.threadsPerBlock

    def L2LoadV1(self):
        return self.basic.blockL2Load / self.lc.threadsPerBlock

    def smL1LoadAllocation(self):
        return self.basic.blockL1LoadAllocation * self.lc.blocksPerSM

    def L1LoadEvicts(self):
        coverage = min(1, max(0, self.device.sizeL1 / self.smL1LoadAllocation()))
        return (self.L1Load() - self.L2LoadV1() )  * (1-coverage)

    def L2LoadV2(self):
        return self.L2LoadV1() + self.L1LoadEvicts()

    def L2Store(self):
        return self.basic.blockL2Store / self.lc.threadsPerBlock


    def memLoadV1(self):
        return self.basic.waveMemLoadISL / self.basic.waveValidCells

    def waveL2Allocation(self):
        return self.basic.waveMemLoadISL - self.basic.waveMemOverlap + self.basic.waveMemStore

    def memLoadOverlap(self):
        coverage = min(1, max(0, (self.device.sizeL2 - self.waveL2Allocation()) / self.basic.waveMemLoadOld))
        return (self.basic.waveMemOverlap * coverage) / self.basic.waveValidCells

    def memLoadV2(self):
        return self.memLoadV1() - self.memLoadOverlap()

    def L2CurrentCoverage(self):
        return min(1, max(0, self.device.sizeL2 / self.waveL2Allocation() ))


    def memLoadEvicts(self):
        return (self.L2LoadV2() - self.memLoadV2()) * (1 - self.L2CurrentCoverage())

    def memLoadV3(self):
        return self.memLoadV2() + self.memLoadEvicts()


    def memStoreV1(self):
        return self.basic.waveMemStore / self.basic.waveValidCells

    def memStoreEvicts(self):
        return max(0, self.L2Store() - self.memStoreV1()) * (1 - self.L2CurrentCoverage())

    def memStoreV2(self):
        return self.memStoreV1() + self.memStoreEvicts()

    def memLoadV4(self):
        return self.memLoadV3() + self.memStoreEvicts()


    def stringKey(self, key, labelWidth, valueWidth):
        kB = key == "smL1LoadAllocation" or key == "waveL2Allocation"
        if not getattr(self, key, None) is None:
            string = "{:{labelWidth}}: {:{valueWidth}.{prec}f} {:2}   ".format(str(key),  getattr(self, key)() / (1024 if kB else 1), "kB" if kB else "",
                                                                               labelWidth=labelWidth, valueWidth=valueWidth, prec= 0 if kB else 1)
        else:
            string = "{:{width}}".format(" ", width=labelWidth+valueWidth+8)
        return string

    def __str__(self):
        columns = [["L1Cycles", "L1Load", "smL1LoadAllocation", "L1LoadEvicts", "L2LoadV1", "L2LoadV2" ],
                   ["memLoadOverlap", "memLoadEvicts", "memLoadV1", "memLoadV2", "memLoadV3", "memLoadV4"],
                   ["waveL2Allocation", "L2CurrentCoverage", "memStoreEvicts", "L2Store", "memStoreV1", "memStoreV2"]]

        return columnPrint(self, columns)

