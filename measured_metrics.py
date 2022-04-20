#!/usr/bin/env python3
from meas_utils import measureMetrics, benchKernel
from column_print import *


class MeasuredMetrics:
    def measure(runFunc, lc):
        self = MeasuredMetrics()
        (
            self.memLoad,
            self.memStore,
            #self.L2Load,
            #self.L2Store,
            self.L2Load_tex,
            self.L2Store,
            #self.L2tex,
            #self.L2ltc,
            #self.L2total,
            #self.L2tagRequests,
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

    def stringKey(self, key, labelWidth, valueWidth):
        kB = False
        if key in self.__dict__:
            string = "{:{labelWidth}}: {:{valueWidth}.1f}{:2} ".format(
                str(key),
                self.__dict__[key] / (1024 if kB else 1),
                "kB" if kB else "",
                labelWidth=labelWidth,
                valueWidth=valueWidth,
            )
        else:
            string = "{:{width}}".format(" ", width=labelWidth + valueWidth)
        return string

    def __str__(self):
        columns = [
            ["L2Load"],
            ["L2Store"],
            ["memLoad"],
            ["memStore"],
            ["lups"],
            ["L1Wavefronts"],
            ["L1Volume"],
        ]

        return columnPrint(self, columns)


class ResultComparer:
    def __init__(self, meas, pred):
        self.meas = meas
        self.pred = pred

    def stringKey(self, key, labelWidth, valueWidth):

        if key.startswith("meas"):
            return self.meas.stringKey(key[4:], labelWidth, valueWidth)
        else:
            return self.pred.stringKey(key, labelWidth, valueWidth)

        #     value = self.meas.__dict__[key[4:]]if key[4:] in self.meas.__dict__ else None
        # else:
        #     value = getattr(self.pred, key)() if not getattr(self.pred, key, None) is None else None
        # if not value is None:
        #     return "{:{labelWidth}}: {:{valueWidth}.1f} {:2}   ".format(str(key), value / (1024 if kB else 1), "kB" if kB else "",
        #                                                                        labelWidth=labelWidth, valueWidth=valueWidth, prec= 0 if kB else 1)
        # else:
        #     return "{:{width}}".format(" ", width=labelWidth+valueWidth+8)

    def __str__(self):
        columns = [
            [
                "L1Cycles",
                "L1Load",
                "smL1Alloc",
                "L1LoadEvicts",
                "L2LoadV1",
                "L2LoadV2",
                "measL2Load_tex",
            ],
            [
                "memLoadOverlap[0]",
                "memLoadEvicts",
                "memLoadV1",
                "memLoadV2",
                "memLoadV3",
                "memLoadV4",
                "measmemLoad",
            ],
            ["waveL2Alloc", "L2NewCoverage", "L2Store", "measL2Store"],
            ["memStoreEvicts", "memStoreV1", "memStoreV2", "measmemStore"],
            ["perfPheno", "perfV1", "perfV2", "perfV3", "perfV4", "measlups"],
        ]

        return columnPrint(self, columns)
