#!/usr/bin/env python3

from warpspeedkernel import *
from griditeration import *
from volumes_isl import *
from volumes_isl3d import *
from column_print import *
from predict import predictPerformance
import math


def perFlop(obj, metric):
    return getattr(obj, metric) / obj.flopsPerLup


def perLup(obj, metric):
    return getattr(obj, metric)


def selectDevice(name):
    if "V100" in name:
        return DeviceVolta()
    if "A100" in name:
        return DeviceAmpere()
    if "2080" in name:
        return Device2080Ti()


class Device:
    def peakFP32(self):
        return self.fp32CycleSM * self.clock * self.smCount * 2


class DeviceVolta(Device):
    CLAllocationSize = 128
    L2FetchSize = 32

    fp64CycleSM = 32
    fp32CycleSM = 64

    API = "CUDA"
    L1Model = "NV"


class DeviceCDNA(Device):
    CLAllocationSize = 64
    L2FetchSize = 64

    fp64CycleSM = 32
    fp32CycleSM = 64

    sizeL1 = 16 * 1024

    API = "HIP"
    L1Model = "CDNA"


class DeviceV100(DeviceVolta):
    clock = 1.38
    smCount = 80
    sizeL2 = 6 * 1024 * 1024
    sizeL1 = 128 * 1024

    L2BW = 2500
    memBW = 790

    name = "V100"


class DeviceAmpere(DeviceVolta):
    clock = 1.410
    smCount = 108
    sizeL2 = 20 * 1024 * 1024
    sizeL1 = 192 * 1024

    L2BW = 4500
    memBW = 1400

    name = "A100"


class DeviceAmpereA100_80GB(DeviceAmpere):
    memBW = 1600
    L2BW = 4500
    name = "A100_80GB"


class Device2080Ti(DeviceVolta):  # unverified
    clock = 2100
    smCount = 68
    sizeL2 = 5.5 * 1024 * 1024
    sizeL1 = 64 * 1024

    L2BW = 2000
    memBW = 590

    name = "2080Ti"


class DeviceMI100(DeviceCDNA):
    clock = 1.2
    smCount = 110
    sizeL2 = 6 * 1024 * 1024

    L2BW = 2500
    memBW = 1200

    name = "MI100"


class DeviceMI210(DeviceCDNA):
    clock = 1.7
    smCount = 104
    sizeL2 = 6 * 1024 * 1024

    fp64CycleSM = 64

    L2BW = 5000
    memBW = 1400

    name = "MI210"


class LaunchConfig:
    def compute(
        kernel,
        block,
        domain,
        blocking_factors,
        device,
        buffers,
        alignmentBytes=0,
    ):
        self = LaunchConfig()
        self.block = block
        self.grid = tuple(
            (domain[i] - 1) // (block[i] * blocking_factors[i]) + 1
            for i in range(len(block))
        )
        self.domain = list(domain)
        self.blocking_factors = blocking_factors
        self.device = device.name
        self.API = device.API

        self.blocksPerSM = predict.getBlocksPerSM(block, kernel.registers)
        if self.blocksPerSM == 0:
            print(
                "Warning: configuration with "
                + str(block[0] * block[1] * block[2])
                + " threads per block and "
                + str(kernel.registers)
                + " cannot be executed"
            )
        self.waveSize = predict.getConcurrentGrid(
            self.blocksPerSM * device.smCount, self.grid
        )

        self.truncatedWaveSize = tuple(min(4, c) for c in self.waveSize)
        self.threadsPerBlock = block[0] * block[1] * block[2]
        self.lupsPerThread = reduce(mul, blocking_factors)
        self.flops = kernel.flops
        self.buffers = buffers
        self.alignmentBytes = alignmentBytes
        self.lupCount = domain[0] * domain[1] * domain[2]
        return self

    def fromDict(values):
        self = LaunchConfig()
        self.__dict__ = values
        return self

    def __str__(self):
        columns = [
            [("block", ""), ("grid", ""), ("waveSize", ""), ("truncatedWaveSize", "")],
            [("blocking_factors", ""), ("threadsPerBlock", ""), ("blocksPerSM", "")],
        ]
        return columnPrint(self, columns)


class BasicMetrics:
    def compute(lc, device, kernel):
        self = BasicMetrics()

        self.L1FieldCycles = getL1Cycles(
            lc.block,
            lc.truncatedWaveSize,
            kernel.loadFields + kernel.storeFields,
            device.L1Model,
        )
        self.L1DataPipeCycles, self.L1TagCycles, self.L1Cycles = self.L1FieldCycles[
            "total"
        ]

        # linearLoadAddresses = [ l.linearAddresses for l in kernel.loadFields  ]
        # linearStoreAddresses = [ l.linearAddresses for l in kernel.storeFields  ]

        self.blockL1LoadAlloc = max(
            1,
            getL1AllocatedLoadBlockVolume(
                lc.block,
                lc.truncatedWaveSize,
                kernel.loadFields,
                device.CLAllocationSize,
            ),
        )
        self.blockL1Load = max(
            1,
            getL2StoreBlockVolume(
                lc.block, lc.truncatedWaveSize, kernel.loadFields, device.L2FetchSize
            ),
        )
        self.warpL1Load = max(
            1, getL1WarpLoadVolume(lc.block, kernel.loadFields, device.L2FetchSize)
        )
        self.blockL2Load = max(
            1,
            getL2LoadBlockVolume(
                lc.block, lc.truncatedWaveSize, kernel.loadFields, device.L2FetchSize
            ),
        )

        self.blockL1TLBPages = max(
            1,
            getL1TLBPages(
                lc.block,
                lc.truncatedWaveSize,
                kernel.loadFields + kernel.storeFields,
                512 * 1024,
            ),
        )

        self.blockL2Store = max(
            1,
            getL2StoreBlockVolume(
                lc.block, lc.truncatedWaveSize, kernel.storeFields, device.L2FetchSize
            ),
        )
        # self.waveMemLoadISL, self.waveMemLoadOld, self.waveMemOverlap, self.waveValidCells = getMemLoadBlockVolumeISL(lc.block, lc.waveSize, lc.grid, kernel.genLoadExprs(), [0,0,0] + lc.domain)
        # self.waveMemStoreISL, self.waveMemStoreOld, self.waveMemStoreOverlap, self.waveValidCells = getMemLoadBlockVolumeISL(lc.block, lc.waveSize, lc.grid, kernel.genStoreExprs(), [0,0,0] + lc.domain)

        (
            self.waveMemLoadNew,
            self.waveMemStoreNew,
            self.waveMemOld,
            self.waveMemLoadOverlap,
            self.waveMemStoreOverlap,
            self.waveValidCells,
        ) = getMemBlockVolumeISL3D(
            kernel.loadFields,
            kernel.storeFields,
            device,
            lc.block,
            lc.grid,
            [0, 0, 0] + lc.domain,
            lc.blocksPerSM * device.smCount,
        )

        self.TLBpages = getWaveLoadTLBPages(
            lc.block,
            lc.waveSize,
            kernel.loadFields + kernel.storeFields,
            2 * 1024 * 1024,
        )
        return self

    def fromDict(values):
        self = BasicMetrics()
        self.__dict__ = values
        return self

    def __str__(self):
        columns = [
            [
                ("TLBpages", ""),
                ("L1Cycles", ""),
                ("blockL1LoadAlloc", "kB"),
                ("blockL1Load", "kB"),
            ],
            [
                ("blockL2Load", "kB"),
                ("blockL2Store", "kB"),
                ("waveMemLoadNew", "kB"),
                ("waveMemStoreNew", "kB"),
            ],
            [
                ("waveValidCells", ""),
                ("waveMemLoadOverlap[0]", "kB"),
                ("waveMemLoadOverlap[1]", "kB"),
                ("waveMemOld[1]", "kB"),
            ],
        ]

        return columnPrint(self, columns)

    def html(self):
        highCount = lambda v: "{:.0f}".format(v)
        smallCount = lambda v: "{:.1f}".format(v)
        kiloByte = lambda v: "{:.1f} kB".format(v / 1024)
        manyByte = lambda v: "{:.0f} B".format(v)
        fewByte = lambda v: "{:.1f} B".format(v)

        columns = [
            [
                ("blockL1LoadAlloc", kiloByte),
                ("blockL1Load", manyByte),
                ("warpL1Load", manyByte),
                ("blockL2Load", manyByte),
            ],
            [
                ("validCells", highCount),
                ("waveMemLoadNew", manyByte),
                ("waveMemLoadOld", manyByte),
                ("waveMemLoadOverlap", manyByte),
            ],
            [
                ("waveValidCells", highCount),
                ("L1Cycles", smallCount),
                ("blockL2Store", manyByte),
                ("waveMemStoreNew", manyByte),
            ],
        ]
        return formattedHtmlColumnPrint(self, columns)


class DerivedMetrics:
    def __init__(self, lc, basic, device, meas=None):
        self.lc = lc
        self.basic = basic
        self.device = device
        self.meas = meas

        lupsPerThread = (
            lc.blocking_factors[0] * lc.blocking_factors[1] * lc.blocking_factors[2]
        )
        self.flopsPerLup = lc.flops

        # Pass through of the estimated cycles quantity
        self.L1DataPipeCycles = self.basic.L1DataPipeCycles / lupsPerThread
        self.L1TagCycles = self.basic.L1TagCycles / lupsPerThread
        self.L1Cycles = self.basic.L1Cycles / lupsPerThread

        # Pass through of the estimated cycles quantity
        self.TLBpages = self.basic.TLBpages

        self.L1TLBPages = self.basic.blockL1TLBPages * self.lc.blocksPerSM

        # Remap block quantity to thread balance
        self.L1Load = self.basic.blockL1Load / self.lc.threadsPerBlock / lupsPerThread

        # Remap block quantity to thread balance
        self.L2LoadV1 = self.basic.blockL2Load / self.lc.threadsPerBlock / lupsPerThread

        # Total memory amount per SM allocated by 128B cache lines touched by loads
        self.smL1Alloc = self.basic.blockL1LoadAlloc * self.lc.blocksPerSM

        # Estimate L1 capacity evictions of loads, using coverage as hitrate proxy
        self.L1coverage = self.smL1Alloc / self.device.sizeL1

        self.L1LoadEvicts = (
            (self.L1Load - self.L2LoadV1)
            * 0.73
            * math.exp(-4.6 * math.exp(-0.34 * self.L1coverage))
        )

        # Version 2 of L2Load that includes the load evictions
        self.L2LoadV2 = self.L2LoadV1 + self.L1LoadEvicts

        # Store volume written through to L2 cache
        self.L2Store = self.basic.blockL2Store / self.lc.threadsPerBlock / lupsPerThread

        # memory load volume from memory footprint of current wave
        self.memLoadV1 = (
            self.basic.waveMemLoadNew / self.basic.waveValidCells / lupsPerThread
        )

        # total allocated memory in L2 cache
        self.waveL2Alloc = self.basic.waveMemLoadNew + self.basic.waveMemStoreNew

        self.memLoadOverlap = tuple(
            w / self.basic.waveValidCells / lupsPerThread
            for w in self.basic.waveMemLoadOverlap
        )

        def rover0(Vnew, Vold):
            if Vold == 0:
                oversubscription = 0.1
            else:
                oversubscription = min(10, max(0.01, Vold / (device.sizeL2 - Vnew)))

            return 1.0 * exp(-0.078 * exp(1.03 * (oversubscription - 0.2)))

        def rover1(Vnew, Vold):
            if Vold == 0:
                oversubscription = 0.1
            else:
                oversubscription = min(10, max(0.01, Vold / (device.sizeL2 - Vnew)))
            return 1.0 * exp(-0.0036 * exp(3.97 * oversubscription))

        # compute memory load balance reduced by hits in previous wave

        self.memLoadOverlapHit = (
            self.memLoadOverlap[0] * rover0(0, self.basic.waveMemOld[0]),
            self.memLoadOverlap[1] * rover1(0, self.basic.waveMemOld[1]),
        )
        self.memLoadV2 = (
            self.memLoadV1 - self.memLoadOverlapHit[0] - self.memLoadOverlapHit[1]
        )

        # memory store volume
        self.memStoreV1 = (
            self.basic.waveMemStoreNew / self.basic.waveValidCells / lupsPerThread
        )

        # compute the L2 cache coverage of the current wave's accesses
        self.L2Oversubscription = self.waveL2Alloc / self.device.sizeL2

        # estimate partially written cache lines evicted before completion using L2 current coverage
        self.memStoreEvicts = max(0, self.L2Store - self.memStoreV1) * (
            1 - 1.0 * np.exp(-0.044 * np.exp(0.61 * self.L2Oversubscription))
        )

        # estimate the L2 load evicts using coverage as proxy
        self.memLoadEvicts = (
            (self.L2LoadV2 - self.memLoadV2)
            * 0.5
            * np.exp(-13 * np.exp(-1.8 * self.L2Oversubscription))
        )

        # memory store balance including store evicts
        self.memStoreV2 = self.memStoreV1 + self.memStoreEvicts

        # memory load balance, assuming that a store evict triggers a read from memory
        self.memLoadV3 = self.memLoadV2 + self.memStoreEvicts

        # compute memory load balance including capacity evicts
        self.memLoadV4 = self.memLoadV3 + self.memLoadEvicts

        self.perfFlops = (
            (
                self.device.clock
                * self.device.smCount
                * self.device.fp32CycleSM
                * 2
                / self.lc.flops
            )
            if lc.flops > 0
            else 0
        )

        self.perfL1 = (
            self.device.smCount * self.device.clock * 32 / self.L1Cycles
            if self.L1Cycles != 0
            else 0
        )

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
        self.perfV1, self.limV1 = selectLimiter(
            [self.perfFlops, self.perfL1, self.perfL2V1, self.perfMemV1]
        )

        # roofline style performance estimate with warm L2
        self.perfV2, self.limV2 = selectLimiter(
            [self.perfFlops, self.perfL1, self.perfL2V1, self.perfMemV2]
        )

        # roofline style performance estimate with capacity evicts
        self.perfV3, self.limV3 = selectLimiter(
            [self.perfFlops, self.perfL1, self.perfL2V2, self.perfMemV3]
        )

        # roofline style performance estimate with RFO balances
        self.perfV4, self.limV4 = selectLimiter(
            [self.perfFlops, self.perfL1, self.perfL2V2, self.perfMemV4]
        )

        # naive roofline style performance estimate with RFO balances
        self.perf2LimV4, self.lim2LimV4 = selectLimiter([self.perfMemV4])

        def epm_no_overlap(device, lc, L1Cycles, L2Volume, memVolume):
            waveLups = (
                self.device.smCount
                * self.lc.blocksPerSM
                * self.lc.threadsPerBlock
                * self.lc.lupsPerThread
            )
            threadCycles = (
                +L1Cycles * self.lc.blocksPerSM * self.lc.threadsPerBlock / 32
                + lc.flops * 2
                + device.clock
                * (
                    memVolume * waveLups / device.memBW
                    + L2Volume * waveLups / device.L2BW
                )
            )
            print(
                "Flops: {:4.0f} L1:  {:4.0f}  L2: {:4.0f}   DRAM: {:4.0f}   total: {:4.0f} ".format(
                    lc.flops * 2,
                    L1Cycles,
                    L2Volume * waveLups / device.L2BW * device.clock,
                    memVolume * waveLups / device.memBW * device.clock,
                    threadCycles,
                )
            )
            return waveLups * self.device.clock / threadCycles

        self.perfEPMV3 = predictPerformance(
            device,
            lc,
            self.L1Cycles,
            (self.L2LoadV2 + self.L2Store) * 32,
            (self.memLoadV3 + self.memStoreV2) * 32,
        )

        if not meas is None:
            self.perfMemPheno = self.device.memBW / max(
                0.1, meas.memLoad + meas.memStore
            )
            self.perfL2Pheno = self.device.L2BW / max(
                0.1, meas.L2Load_tex + meas.L2Store
            )

            self.perfL1Pheno = (
                self.device.smCount
                * self.device.clock
                / max(meas.L1DataPipeWavefronts, meas.L1TagWavefronts)
            )
            self.perfPheno, self.limPheno = selectLimiter(
                [self.perfFlops, self.perfL1Pheno, self.perfL2Pheno, self.perfMemPheno]
            )

            self.perf2LimPheno, self.lim2LimPheno = selectLimiter([self.perfMemPheno])
            self.perfEPMPheno = predictPerformance(
                device,
                lc,
                max(meas.L1DataPipeWavefronts, meas.L1TagWavefronts) * 32,
                (meas.L2Load_tex + meas.L2Store) * 32,
                (meas.memLoad + meas.memStore) * 32,
            )

        if getattr(lc, "flops", 0) > 0:
            for a in dir(self):
                if a.startswith("perf"):
                    self.__dict__["perfTFlops" + a[4:]] = (
                        getattr(self, a) * lc.flops / 1000
                    )

    def stringKey(self, key, labelWidth, valueWidth):
        kB = key == "smL1Alloc" or key == "waveL2Alloc"
        if key in self.__dict__:
            string = "{:{labelWidth}}: {:{valueWidth}.{prec}f}{:2} ".format(
                str(key),
                self.__dict__[key] / (1024 if kB else 1),
                "kB" if kB else "",
                labelWidth=labelWidth,
                valueWidth=valueWidth,
                prec=0 if kB else 1,
            )
        else:
            string = "{:{width}}".format(" ", width=labelWidth + valueWidth + 5)
        return string

    def columns(self):
        columns = [
            [
                ("L1Load", "B/Lup"),
                ("smL1Alloc", "kB"),
                ("L1LoadEvicts", "B/Lup"),
                ("L2LoadV1", "B/Lup"),
                ("L2LoadV2", "B/Lup"),
            ],
            [
                ("memLoadOverlap[0]", "B/Lup"),
                ("memLoadOverlap[1]", "B/Lup"),
                ("memLoadV1", "B/Lup"),
                ("memLoadV2", "B/Lup"),
                ("memLoadV3", "B/Lup"),
            ],
            [
                ("basic.waveMemOld[0]", "MB"),
                ("basic.waveMemOld[1]", "MB"),
                ("L2Store", "B/Lup"),
                ("memStoreV1", "B/Lup"),
                ("memStoreV2", "B/Lup"),
            ],
            [
                ("L1Cycles", "cyc"),
                ("perfFlops", "GFlop/s"),
                ("perfMemV3", "GFlop/s"),
                ("perfL2V2", "GFlop/s"),
                ("perfL1", "GFlop/s"),
                ("perfV3", "GFlop/s"),
            ],
        ]
        return columns

    def __str__(self):
        return columnPrint(self, self.columns())

    def html(self):
        return htmlColumnPrint(self, self.columns())
