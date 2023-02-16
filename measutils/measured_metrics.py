#!/usr/bin/env python3
import stencil_runner.stencil_runner as stencil_runner
from column_print import *
from pystencils.display_utils import get_code_str


class MeasuredMetrics:
    def measure(kernel, lc):

        codeText = (
            get_code_str(kernel)
            .replace("FUNC_PREFIX", '#define int64_t size_t \n extern "C" __global__ ')
            .replace("RESTRICT", "__restrict__")
        )
        self = MeasuredMetrics()
        (
            self.memLoad,
            self.memStore,
            # self.L2Load_tex,
            # self.L2Store,
            # self.L1Wavefronts,
        ) = stencil_runner.measureMetrics2BufferKernel(
            lc.API,
            ["dram_load_bytes", "dram_write_bytes"],
            codeText,
            "kernel",
            lc.block,
            lc.grid,
            lc.bufferSizeBytes,
        )

        self.memLoad *= 1024.0 / (lc.domain[0] * lc.domain[1] * lc.domain[2])
        self.memStore *= 1024.0 / (lc.domain[0] * lc.domain[1] * lc.domain[2])

        self.L2Load_tex = 222;
        self.L2Store = 333;
        self.L1Wavefronts = 444;

        self.L2ltc = 1
        self.L2total = 1

        time = stencil_runner.time2BufferKernel(
            lc.API, codeText, "kernel", lc.block, lc.grid, lc.bufferSizeBytes
        )
        self.lups = lc.domain[0] * lc.domain[1] * lc.domain[2] / time / 1e9

        return self

    def fromDict(values):
        self = MeasuredMetrics()
        self.__dict__ = values
        return self

    def __str__(self):
        columns = [
            [
                ("L2Load_tex", "B"),
                ("L2Store", "B"),
                ("memLoad", "B"),
                ("memStore", "B"),
            ],
            [("L1Wavefronts", ""), ("lups", "Lup/s")],
        ]

        return columnPrint(self, columns)


class ResultComparer:
    def __init__(self, meas, pred):
        # for v in vars(meas):
        #    setattr(self, v, getattr(meas, v))

        # for v in vars(pred):
        #    setattr(self, v, getattr(pred, v))

        self.m = meas
        self.p = pred

    def columns(self):
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
                ("p.perfL1", "GFlop/s"),
                ("p.perfL2V2", "GFlop/s"),
                ("p.perfMemV4", "GFlop/s"),
                ("p.perfV4", "GFlop/s"),
                ("p.perfPheno", "GFlop/s"),
                ("m.lups", "GFlop/s"),
                ("p.limPheno", ""),
            ],
        ]
        return columns

    def __str__(self):
        return columnPrint(self, self.columns())

    def html(self):
        return htmlColumnPrint(self, self.columns())

    def printSASS(code):
        print(code)
        cubin = pycuda.compiler.compile(
            code, options=["-w", "-std=c++11"], arch="sm_70"
        )

        run(['echo "' + code + '" >> temp.cubin'], stdout=PIPE, shell=True)

        newFile = open("temp.cubin", "wb")
        newFile.write(cubin)
        newFile.close()

        result = run(["nvdisasm  temp.cubin"], stdout=PIPE, shell=True)

        print(len(result.stdout.decode("utf-8").split("\n")))

        print(result.stdout.decode("utf-8"))

        newFile = open("temp.disasm", "wb")
        newFile.write(result.stdout)
        newFile.close()
