#!/usr/bin/env python3

from warpspeedkernel import *
from griditeration import *
from volumes_isl import *
from volumes_isl3d import *
from column_print import *
from predict import predictPerformance
from devices import *
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

        if device.L1Model == "CDNA":
            kernel.fuseAccesses()

        self.fieldL1Cycles = getL1Cycles(
            lc.block,
            lc.truncatedWaveSize,
            kernel.loadFields + kernel.storeFields,
            device,
        )

        self.L1DataPipeCycles, self.L1TagCycles, self.L1Cycles = self.fieldL1Cycles[
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
                lc.block, lc.truncatedWaveSize, kernel.loadFields, device
            )["total"],
        )

        self.warpL1Load = max(
            1, getL1WarpLoadVolume(lc.block, kernel.loadFields, device.CLFetchSize)
        )

        self.fieldBlockL2Load = getL2LoadBlockVolume(
            lc.block, lc.truncatedWaveSize, kernel.loadFields, device.CLFetchSize
        )
        self.blockL2Load = max(1, self.fieldBlockL2Load["total"])

        self.fieldBlockL2LoadOverlap = getL2LoadOverlapBlockVolume(
            lc.block, lc.grid, kernel.loadFields, device.CLFetchSize
        )
        # self.blockL2Load = max(1, self.fieldBlockL2Load["total"])

        self.blockL1TLBPages = max(
            1,
            getL1TLBPages(
                lc.block,
                lc.truncatedWaveSize,
                kernel.loadFields + kernel.storeFields,
                512 * 1024,
            ),
        )

        self.fieldBlockL2Store = getL2StoreBlockVolume(
            lc.block, lc.truncatedWaveSize, kernel.storeFields, device
        )

        self.blockL2Store = self.fieldBlockL2Store["total"]

        (
            self.waveMemLoadNew,
            self.waveMemStoreNew,
            self.waveMemOld,
            self.waveMemLoadOverlap,
            self.waveMemStoreOverlap,
            self.waveValidCells,
            self.fieldWaveMemVolumes,
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
        self.fieldL1Cycles = {
            fieldName: tuple(c / lupsPerThread for c in cycles)
            for (fieldName, cycles) in basic.fieldL1Cycles.items()
        }

        self.L1DataPipeCycles = self.fieldL1Cycles["total"][0]
        self.L1TagCycles = self.fieldL1Cycles["total"][1]
        self.L1Cycles = self.fieldL1Cycles["total"][2]

        # Pass through of the estimated cycles quantity
        self.TLBpages = self.basic.TLBpages

        self.L1TLBPages = self.basic.blockL1TLBPages * self.lc.blocksPerSM

        # Total memory amount per SM allocated by 128B cache lines touched by loads
        # Multiplying with the number of blocks on the SM seems logic, but blocks are in different phases
        # and don't allocate the memory at the same time
        self.smL1Alloc = self.basic.blockL1LoadAlloc  # *  self.lc.blocksPerSM

        # Estimate L1 capacity evictions of loads, using coverage as hitrate proxy
        self.L1coverage = self.smL1Alloc / self.device.sizeL1

        # Remap block quantity to thread balance
        self.L1Load = self.basic.blockL1Load / self.lc.threadsPerBlock / lupsPerThread

        self.L1OverlapOversubscription = min(
            5, self.basic.fieldBlockL2Load["total"] / self.device.sizeL1
        )

        # Remap block quantity to thread balance
        self.fieldL2Load = {
            fieldName: (
                self.basic.fieldBlockL2Load[fieldName]
                - (
                    self.basic.fieldBlockL2LoadOverlap[fieldName]
                    * (exp(-0.0036 * exp(8 * self.L1OverlapOversubscription)))
                )
            )
            / self.lc.threadsPerBlock
            / self.lc.lupsPerThread
            for (fieldName) in self.basic.fieldBlockL2Load.keys()
        }
        print(exp(-0.0036 * exp(3.97 * self.L1OverlapOversubscription)))
        print(self.basic.fieldBlockL2Load["total"])
        print(self.basic.fieldBlockL2LoadOverlap["total"])
        print(self.L1OverlapOversubscription)
        print(self.fieldL2Load)

        self.L2LoadV1 = self.fieldL2Load["total"]

        self.L1LoadEvicts = (
            (self.L1Load - self.L2LoadV1)
            * 0.73
            * math.exp(-4.6 * math.exp(-0.34 * self.L1coverage))
        )

        # Version 2 of L2Load that includes the load evictions
        self.L2LoadV2 = self.L2LoadV1 + self.L1LoadEvicts

        # Store volume written through to L2 cache
        self.fieldL2Store = {
            fieldName: v / self.lc.threadsPerBlock / self.lc.lupsPerThread
            for (fieldName, v) in self.basic.fieldBlockL2Store.items()
        }
        self.L2Store = self.fieldL2Store["total"]

        # memory load volume from memory footprint of current wave
        self.memLoadV1 = (
            self.basic.waveMemLoadNew / self.basic.waveValidCells / lupsPerThread
        )

        self.fieldMemLoadV1 = {
            fieldName: v["VNew"] / self.basic.waveValidCells / lupsPerThread
            for (fieldName, v) in self.basic.fieldWaveMemVolumes.items()
        }

        # total allocated memory in L2 cache
        self.waveL2Alloc = self.basic.waveMemLoadNew + self.basic.waveMemStoreNew

        self.memLoadOverlap = tuple(
            w / self.basic.waveValidCells / lupsPerThread
            for w in self.basic.waveMemLoadOverlap
        )

        self.fieldMemLoadOverlap = {
            fieldName: (
                v["VOverlapY"] / self.basic.waveValidCells / lupsPerThread,
                v["VOverlapZ"] / self.basic.waveValidCells / lupsPerThread,
            )
            for (fieldName, v) in self.basic.fieldWaveMemVolumes.items()
        }

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

        self.fieldMemLoadV2 = {
            fieldName: self.fieldMemLoadV1[fieldName]
            - self.fieldMemLoadOverlap[fieldName][0]
            * rover0(0, self.basic.waveMemOld[0])
            - self.fieldMemLoadOverlap[fieldName][1]
            * rover1(0, self.basic.waveMemOld[1])
            for (fieldName, v) in self.basic.fieldWaveMemVolumes.items()
        }
        #
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
            * 0.386
            * np.exp(-9 * np.exp(-0.639 * self.L2Oversubscription))
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

        self.perfL1 = self.device.smCount * self.device.clock / self.L1Cycles

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
