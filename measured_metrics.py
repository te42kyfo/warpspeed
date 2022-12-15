#!/usr/bin/env python3
from meas_utils import measureMetrics, benchKernel
from column_print import *


class MeasuredMetrics:
    def measure(runFunc, lc):
        self = MeasuredMetrics()
        (
            self.memLoad,
            self.memStore,
            # self.L2Load,
            # self.L2Store,
            self.L2Load_tex,
            self.L2Store,
            # self.L2tex,
            # self.L2ltc,
            # self.L2total,
            # self.L2tagRequests,
            self.L1Wavefronts,
        ) = measureMetrics(runFunc, lc.domain)

        self.L2ltc = 1
        self.L2total = 1

        time = benchKernel(runFunc)
        self.lups = lc.domain[0] * lc.domain[1] * lc.domain[2] / time / 1e6
        return self

    def fromDict(values):
        self = MeasuredMetrics()
        self.__dict__ = values
        return self

    # def stringKey(self, key, labelWidth, valueWidth):
    #    kB = False
    #    if key in self.__dict__:
    #        string = "{:{labelWidth}}: {:{valueWidth}.1f}{:2} ".format(
    #            str(key),
    #            self.__dict__[key] / (1024 if kB else 1),
    #            "kB" if kB else "",
    #            labelWidth=labelWidth,
    #            valueWidth=valueWidth,
    #        )
    #    else:
    #        string = "{:{width}}".format(" ", width=labelWidth + valueWidth)
    #    return string

    def __str__(self):
        columns = [
            [
                ("L2Load_tex", "B"),
                ("L2Store", "B"),
                ("memLoad", "B"),
                ("memStore", "B"),
            ],
            [("L1Wavefronts", ""),
             ("lups", "Lup/s")],
        ]

        return columnPrint(self, columns)


class ResultComparer:
    def __init__(self, meas, pred):
        #for v in vars(meas):
        #    setattr(self, v, getattr(meas, v))

        #for v in vars(pred):
        #    setattr(self, v, getattr(pred, v))

        self.m = meas
        self.p = pred

    def __str__(self):
        columns = [
            [
                ("p.L1Cycles", ""),
                ("p.L1Load", "B"),
                ("p.smL1Alloc", "kB"),
                ("p.L1LoadEvicts", "B"),
                ("p.L2LoadV1", "B"),
                ("p.L2LoadV2", "B"),
                ("m.L2Load_tex", "B"),
            ],
            [
                ("p.memLoadV1", "B"),
                ("p.memLoadV2", "B"),
                ("p.memLoadV3", "B"),
                ("p.memLoadV4", "B"),
                ("m.memLoad", "B"),
            ],
            [
                ("p.memLoadOverlap[0]", "B"),
                ("p.memLoadOverlap[1]", "B"),
                ("p.waveL2Alloc", "MB"),
                ("p.memLoadEvicts", "B"),
                ("p.L2Store", "B"),
                ("m.L2Store", "B"),
            ],
            [
                ("p.basic.waveMemOld[0]", "MB"),
                ("p.basic.waveMemOld[1]", "MB"),
                ("p.memStoreEvicts", "B"),
                ("p.memStoreV1", "B"),
                ("p.memStoreV2", "B"),
                ("m.memStore", "B"),
            ],
            [
                ("p.perfL1", "GLup/s"),
                ("p.perfL2V2", "GLup/s"),
                ("p.perfMemV4", "GLup/s"),
                ("p.perfV4", "GLup/s"),
                ("p.perfPheno", "GLup/s"),
                ("m.lups", "GLup/s"),
                ("p.limPheno", ""),
            ],
        ]

        return columnPrint(self, columns)
