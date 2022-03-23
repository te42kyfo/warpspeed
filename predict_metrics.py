#!/usr/bin/env python3

from griditeration import *
from volumes_isl import *
from volumes_isl3d import *
from column_print import *
import math

def selectDevice(name):
    if "V100" in name:
        return DeviceVolta()
    if "A100" in name:
        return DeviceAmpere()
    if "2080" in name:
        return Device2080Ti()

class DeviceVolta:
    def __init__(self):
        self.clock = 1.38
        self.smCount = 80
        self.sizeL2 = 6 * 1024 * 1024
        self.sizeL1 = 128*1024

        self.L2FetchSize = 32

        self.L2BW = 2500
        self.memBW = 790

        self.L2totalBW = 5500
        self.L2texBW = 4500
        self.L2ltcBW = 4500/2
        self.L2TagRate = 40 * self.clock / 1000



        self.name = "V100"


class DeviceAmpere: # unverified
    def __init__(self):
        self.clock = 1.410
        self.smCount = 108
        self.sizeL2 = 20 * 1024 * 1024
        self.sizeL1 = 192*1024

        self.L2FetchSize = 32

        self.L2BW = 4500
        self.memBW = 1400

        self.L2totalBW = 5500
        self.L2texBW = 4500
        self.L2ltcBW = 4500/2

        self.L2TagRate = 40 * self.clock

        self.name = "A100"

class Device2080Ti: # unverified
    def __init__(self):
        self.clock = 2100
        self.smCount = 68
        self.sizeL2 = 5.5 * 1024 * 1024
        self.sizeL1 = 64*1024

        self.L2BW = 2000
        self.memBW = 590

        self.name = "2080Ti"

class LaunchConfig:
    def compute( kernel, block, domain, blocking_factors, device):
        self = LaunchConfig()
        self.block = block
        self.grid = tuple( (domain[i]-1) // block[i] + 1 for i in range(len(block)))
        self.domain = list(domain)
        self.blocking_factors = blocking_factors
        self.device = device.name


        self.blocksPerSM = predict.getBlocksPerSM(block, kernel.registers)

        self.waveSize = predict.getConcurrentGrid(
            self.blocksPerSM * device.smCount, self.grid
        )

        self.truncatedWaveSize = tuple(min(4, c) for c in self.waveSize)
        self.threadsPerBlock = block[0] * block[1] * block[2]
        self.lupsPerThread = reduce(mul, blocking_factors)
        return self

    def fromDict(values):
        self = LaunchConfig()
        self.__dict__ = values
        return self


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
    def compute(lc, device, kernel):
        self = BasicMetrics()
        self.L1Cycles = getL1Cycles(lc.block, lc.truncatedWaveSize, {**kernel.genLoads(), **kernel.genStores()})
        self.blockL1LoadAlloc = max(1, getL1AllocatedLoadBlockVolume(lc.block, lc.truncatedWaveSize, kernel.genLoads()))
        self.blockL1Load = max(1, getL2StoreBlockVolume(lc.block, lc.truncatedWaveSize, kernel.genLoads()))
        self.warpL1Load = max(1, getL1WarpLoadVolume(lc.block, kernel.genLoads()))
        self.blockL2Load =  max(1, getL2LoadBlockVolume(lc.block, lc.truncatedWaveSize, kernel.genLoads(), device.L2FetchSize))
        self.blockL2Load32 =  max(1, getL2LoadBlockVolume(lc.block, lc.truncatedWaveSize, kernel.genLoads(), 32))
        self.blockL2Load64 =  max(1, getL2LoadBlockVolume(lc.block, lc.truncatedWaveSize, kernel.genLoads(), 64))
        self.blockL2Store =  max(1, getL2StoreBlockVolume(lc.block, lc.truncatedWaveSize, kernel.genStores()))
        self.waveMemLoad = max(1, getMemLoadWaveVolume(lc.block, lc.waveSize, kernel.genLoads()))
        self.waveMemStore = max(1, getMemStoreWaveVolume(lc.block, lc.waveSize, kernel.genStores()))
        self.waveMemLoadISL, self.waveMemLoadOld, self.waveMemOverlap, self.waveValidCells = getMemLoadBlockVolumeISL(lc.block, lc.waveSize, lc.grid, kernel.genLoadExprs(), [0,0,0] + lc.domain)
        self.waveMemStoreISL, self.waveMemStoreOld, self.waveMemStoreOverlap, self.waveValidCells = getMemLoadBlockVolumeISL(lc.block, lc.waveSize, lc.grid, kernel.genStoreExprs(), [0,0,0] + lc.domain)
        self.waveMemLoadOld = max(1, self.waveMemLoadOld)

        self.waveMemNewISL3D, self.waveMemOldISL3D, self.waveMemOverlapISL3D, self.waveValidCellsISL3D = getMemBlockVolumeISL3D({**kernel.getLoadExprs3D(), **kernel.getStoreExprs3D()}, device, lc.block, lc.grid, [0,0,0] + lc.domain, lc.blocksPerSM * device.smCount)
        deviceA40MB = DeviceAmpere()
        deviceA40MB.sizeL2 = 40 * 1024 * 1024
        self.waveMemNewISL3D40MB, self.waveMemOldISL3D40MB, self.waveMemOverlapISL3D40MB, self.waveValidCellsISL3D40MB = getMemBlockVolumeISL3D({**kernel.getLoadExprs3D(), **kernel.getStoreExprs3D()}, deviceA40MB, lc.block, lc.grid, [0,0,0] + lc.domain, lc.blocksPerSM * device.smCount)
        self.TLBpages = getWaveLoadTLBPages(lc.block, lc.waveSize, {**kernel.genLoads(), **kernel.genStores()}, 2*1024*1024)
        return self

    def fromDict(values):
        self = BasicMetrics(0)
        self.__dict__ = values
        return self

    def stringKey(self, key, labelWidth, valueWidth):
        kB = key != "L1Cycles" and key != "waveValidCells" and key != "TLBpages"
        if key in self.__dict__:
            string = "{:{labelWidth}}: {:{valueWidth}.0f} {:2}   ".format(str(key),  self.__dict__[key] / (1024 if kB else 1), "kB" if kB else "",
                                                                          labelWidth=labelWidth, valueWidth=valueWidth)
        else:
            string = "{:{width}}".format(" ", width=labelWidth+valueWidth)
        return string

    def __str__(self):
        columns = [["blockL1LoadAlloc", "blockL1Load", "warpL1Load", "blockL2Load"],
                   ["waveMemLoad", "waveMemLoadISL", "waveMemLoadOld", "waveMemOverlap"],
                   ["waveValidCells", "L1Cycles", "blockL2Store", "waveMemStoreISL"],
                   ["TLBpages"]]
        return columnPrint(self, columns)


    def html(self):

        htmlString = "<table><tr>"

        highCount = lambda v : "{:.0f}".format(v)
        smallCount = lambda v : "{:.1f}".format(v)
        kiloByte = lambda v : "{:.1f} kB".format(v / 1024)
        manyByte = lambda v : "{:.0f} B".format(v)
        fewByte = lambda v : "{:.1f} B".format(v)

        columns = [[("blockL1LoadAlloc", kiloByte),
                    ("blockL1Load", manyByte),
                    ("warpL1Load", manyByte),
                    ("blockL2Load", manyByte)],
                   [("waveMemLoad", manyByte),
                    ("waveMemLoadISL", manyByte),
                    ("waveMemLoadOld", manyByte),
                    ("waveMemOverlap", manyByte)],
                   [("waveValidCells", highCount),
                    ("L1Cycles", smallCount),
                    ("blockL2Store", manyByte),
                    ("waveMemStore", manyByte)]
                   ]
        return formattedHtmlColumnPrint(self, columns)



class DerivedMetrics:
    def __init__(self, lc, basic, device, meas=None):
        self.lc = lc
        self.basic = basic
        self.device = device
        self.meas = meas

        lupsPerThread = lc.blocking_factors[0] * lc.blocking_factors[1] * lc.blocking_factors[2]

        # Pass through of the estimated cycles quantity
        self.L1Cycles = self.basic.L1Cycles / lupsPerThread


        # Pass through of the estimated cycles quantity
        self.TLBpages = self.basic.TLBpages

        # Remap block quantity to thread balance
        self.L1Load = self.basic.blockL1Load / self.lc.threadsPerBlock / lupsPerThread

        # Remap block quantity to thread balance
        self.L2LoadV1 = self.basic.blockL2Load / self.lc.threadsPerBlock / lupsPerThread

        # Total memory amount per SM allocated by 128B cache lines touched by loads
        self.smL1Alloc = self.basic.blockL1LoadAlloc * self.lc.blocksPerSM

        # Estimate L1 capacity evictions of loads, using coverage as hitrate proxy
        self.L1coverage = self.smL1Alloc / self.device.sizeL1

        self.L1LoadEvicts = (self.L1Load - self.L2LoadV1 )  * 0.73*math.exp(-4.6*math.exp(-0.34*self.L1coverage))

        # Version 2 of L2Load that includes the load evictions
        self.L2LoadV2 = self.L2LoadV1 + self.L1LoadEvicts

        # Store volume written through to L2 cache
        self.L2Store = self.basic.blockL2Store / self.lc.threadsPerBlock / lupsPerThread

        # memory load volume from memory footprint of current wave
        self.memLoadV1 = self.basic.waveMemLoadISL / self.basic.waveValidCells / lupsPerThread

        # total allocated memory in L2 cache
        self.waveL2Alloc = self.memLoadV1 * self.lc.threadsPerBlock * self.lc.blocksPerSM * self.device.smCount * lupsPerThread + self.basic.waveMemStoreISL

        # computing the balance reduction from overlapping/hits in the previous wave. Use coverage of previous wave with remaining L2 capacity.
        self.memLoadOverlapAbsolute = self.basic.waveMemOverlap / self.basic.waveValidCells / lupsPerThread
        self.memLoadCoverage = (self.device.sizeL2 - self.waveL2Alloc) / self.basic.waveMemLoadOld if self.basic.waveMemLoadOld != 0 else 1
        self.memLoadOverlap = (self.basic.waveMemOverlap * 1.0 * np.exp(-0.6*np.exp(-2.4*self.memLoadCoverage))) / self.basic.waveValidCells / lupsPerThread

        # compute memory load balance reduced by hits in previous wave
        self.memLoadV2 = self.memLoadV1 - self.memLoadOverlap

        # compute the L2 cache coverage of the current wave's accesses
        self.L2Oversubscription = self.waveL2Alloc / self.device.sizeL2

                # estimate the L2 load evicts using coverage as proxy
        self.memLoadEvicts = (self.L2LoadV2 - self.memLoadV2) * 0.5 * np.exp(-13*np.exp(-1.8*self.L2Oversubscription))

        # compute memory load balance including capacity evicts
        self.memLoadV3 = self.memLoadV2 + self.memLoadEvicts

        # memory store volume
        self.memStoreV1 = self.basic.waveMemStoreISL / self.basic.waveValidCells / lupsPerThread

        # estimate partially written cache lines evicted before completion using L2 current coverage
        self.memStoreEvicts = max(0, self.L2Store - self.memStoreV1) * 0.66 * np.exp(-3.4*np.exp(-2.3* self.L2Oversubscription))

        # memory store balance including store evicts
        self.memStoreV2 = self.memStoreV1 + self.memStoreEvicts

        # memory load balance, assuming that a store evict triggers a read from memory
        self.memLoadV4 = self.memLoadV3 + self.memStoreEvicts



        self.memAll = (self.basic.waveMemNewISL3D - self.basic.waveMemOverlapISL3D)  / self.basic.waveValidCellsISL3D
        self.memAll40MB = (self.basic.waveMemNewISL3D40MB - self.basic.waveMemOverlapISL3D40MB)  / self.basic.waveValidCellsISL3D40MB


        self.perfL1 = self.device.smCount * self.device.clock * 32 / self.L1Cycles if self.L1Cycles != 0 else 0

        # L2 bandwidth performance estimate. V1 without L1 evicts. load and store are independent
        self.perfL2V1 = self.device.L2BW / (self.L2LoadV1 + self.L2Store)

        # L2 bandwidth performance estimate. V2 with L1 evicts. load and store are independent
        self.perfL2V2 = self.device.L2BW / (self.L2LoadV2 + self.L2Store)

        # memory bandwidth performance estimate. V1 with simple footprints
        self.perfMemV1 = self.device.memBW / (self.memLoadV1 + self.memStoreV1)

        # memory bandwidth performance estimate. V2 with warm L2 cache
        self.perfMemV2 = self.device.memBW / (self.memLoadV2 + self.memStoreV1)

        # memory bandwidth performance estimate. V3 with warm L2 cache and capacity evicts
        self.perfMemV3 = self.device.memBW / (self.memLoadV3 + self.memStoreV2)

        # memory bandwidth performance estimate. V4 with additional RFO balance
        self.perfMemV4 = self.device.memBW / (self.memLoadV4 + self.memStoreV2)

        def selectLimiter(limiters):
            lowest = limiters[0]
            limiter = 0
            for l in range(len(limiters)):
                if limiters[l] < lowest:
                    lowest = limiters[l]
                    limiter = l
            return lowest, limiter

        # roofline style performance estimate
        self.perfV1, self.limV1 = selectLimiter([self.perfL1, self.perfL2V1, self.perfMemV1])

        # roofline style performance estimate with warm L2
        self.perfV2, self.limV2 = selectLimiter([self.perfL1, self.perfL2V1, self.perfMemV2])

        # roofline style performance estimate with capacity evicts
        self.perfV3, self.limV3 = selectLimiter([self.perfL1, self.perfL2V2, self.perfMemV3])

        # roofline style performance estimate with RFO balances
        self.perfV4, self.limV4 = selectLimiter([self.perfL1, self.perfL2V2, self.perfMemV4])

        if not meas is None:
            self.perfMemPheno = self.device.memBW / (meas.memLoad  + meas.memStore)
            self.perfL2Pheno = min(self.device.L2BW / meas.L2Load_tex, self.device.L2BW / meas.L2Store)
            self.perfPheno, self.limPheno = selectLimiter([self.perfL1, self.perfL2Pheno, self.perfMemPheno])
            self.perfPhenoL2Ext, self.limPhenoL2Ext = selectLimiter(
                [self.device.L2totalBW / meas.L2total, self.device.L2texBW / meas.L2tex, self.device.L2ltcBW / meas.L2ltc])

            self.perfPhenoL2Tag = self.device.L2TagRate / meas.L2tagRequests

    def stringKey(self, key, labelWidth, valueWidth):
        kB = key == "smL1Alloc" or key == "waveL2Alloc"
        if key in self.__dict__:
            string = "{:{labelWidth}}: {:{valueWidth}.{prec}f}{:2} ".format(str(key),  self.__dict__[key] / (1024 if kB else 1), "kB" if kB else "",
                                                                               labelWidth=labelWidth, valueWidth=valueWidth, prec= 0 if kB else 1)
        else:
            string = "{:{width}}".format(" ", width=labelWidth+valueWidth+5)
        return string


    def __str__(self):
        columns = [["L1Cycles", "L1Load", "smL1Alloc", "L1LoadEvicts", "L2LoadV1", "L2LoadV2"],
                   ["memLoadOverlap", "memLoadEvicts", "memLoadV1", "memLoadV2", "memLoadV3", "memLoadV4"],
                   ["waveL2Alloc", "L2Oversubscription", "memStoreEvicts", "L2Store", "memStoreV1", "memStoreV2"]]

        return columnPrint(self, columns)

    def html(self):

        htmlString = "<table><tr>"

        highCount = lambda v : "{:.0f}".format(v)
        smallCount = lambda v : "{:.1f}".format(v)
        kiloByte = lambda v : "{:.1f} kB".format(v / 1024)
        manyByte = lambda v : "{:.0f} B".format(v)
        fewByte = lambda v : "{:.1f} B".format(v)
        gflops = lambda v : "{:.1f} GFlop/s".format(v)

        columns = [[("L1Cycles", smallCount),
                    ("L1Load", fewByte),
                    ("smL1Alloc", kiloByte),
                    ("L1LoadEvicts", fewByte),
                    ("L2LoadV1", fewByte),
                    ("L2LoadV2", fewByte)],
                   [("memLoadOverlap", smallCount),
                    ("memLoadEvicts", fewByte),
                    ("memLoadV1", fewByte),
                    ("memLoadV2", fewByte),
                    ("memLoadV3", fewByte),
                    ("memLoadV4", fewByte)],
                   [("waveL2Alloc", kiloByte),
                    ("L2Oversubscription", smallCount),
                    ("memStoreEvicts", fewByte),
                    ("L2Store", fewByte),
                    ("memStoreV1", fewByte),
                    ("memStoreV2", fewByte)],
                   [("perfV1", gflops),
                    ("perfV2", gflops),
                    ("perfV3", gflops),
                    ("perfV4", gflops)]
                   ]
        return formattedHtmlColumnPrint(self, columns)
