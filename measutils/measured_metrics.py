#!/usr/bin/env python3
from stencil_runner.stencil_runner import *
from column_print import *


class MeasuredMetrics:
    def measure(codeText, lc):
        codeText = (
            codeText.replace(
                "FUNC_PREFIX", '#define int64_t size_t \n extern "C" __global__ '
            ).replace("RESTRICT", "__restrict__")
            # .replace("blockIdx.x", "0")
            # .replace("blockIdx.y", "0")
            # .replace("blockIdx.z", "0")
            # .replace("threadIdx.x", "(threadIdx.x%64)")
            # .replace("threadIdx.y", "1")
            # .replace("threadIdx.z", "1")
        )

        # print(codeText)

        self = MeasuredMetrics()

        time = timeNBufferKernel(
            lc.API,
            codeText,
            "kernel",
            lc.block,
            lc.grid,
            lc.buffers,
            lc.alignmentBytes,
        )

        if lc.API == "HIP":
            (
                self.memLoad,
                self.memStore,
            ) = measureMetricsNBufferKernel(
                lc.API,
                ["FETCH_SIZE", "WRITE_SIZE"],
                codeText,
                "kernel",
                lc.block,
                lc.grid,
                lc.buffers,
                lc.alignmentBytes,
            )

            (
                self.L2Load_tex,
                self.L2Store,
                self.L1Wavefronts_TD,
                self.L1Wavefronts_TA,
                self.UTCL1_requests,
                self.UTCL1_miss,
            ) = measureMetricsNBufferKernel(
                lc.API,
                [
                    "TCP_TCC_READ_REQ_sum",
                    "TCP_TCC_WRITE_REQ_sum",
                    "TD_TD_BUSY_sum",
                    "TA_TA_BUSY_sum",
                    "TCP_UTCL1_REQUEST_sum",
                    "TCP_UTCL1_TRANSLATION_MISS_sum",
                ],
                codeText,
                "kernel",
                lc.block,
                lc.grid,
                lc.buffers,
                lc.alignmentBytes,
            )

            self.memLoad *= 1024.0 / lc.lupCount
            self.memStore *= 1024.0 / lc.lupCount

            self.L2Load_tex *= 64.0 / lc.lupCount
            self.L2Load = self.L2Load_tex
            self.L2Store *= 64.0 / lc.lupCount

            self.L1Wavefronts_TA *= 1.0 / lc.lupCount
            self.L1Wavefronts_TD *= 1.0 / lc.lupCount
            self.L1TagWavefronts = self.L1Wavefronts_TA
            self.L1DataPipeWavefronts = self.L1Wavefronts_TD

            self.UTCL1_requests *= 1.0 / lc.lupCount
            self.UTCL1_miss *= 1.0 / lc.lupCount
            self.L1Wavefronts = self.L1Wavefronts_TA
        else:
            (
                self.memLoad,
                self.memStore,
                self.L2Load,
                self.L2Store,
                self.L2Load_tex,
                self.L2Store_tex,
                self.L2tex,
                # self.L2ltc,
                # self.L2total,
                self.L2tagRequests,
                self.L1DataPipeWavefronts,
                self.L1TagWavefronts,
                self.L1CyclesActive,
            ) = measureMetricsNBufferKernel(
                lc.API,
                [
                    "dram__bytes_read.sum",
                    "dram__bytes_write.sum",
                    "lts__t_sectors_op_read.sum",
                    "lts__t_sectors_op_write.sum",
                    "lts__t_sectors_srcunit_tex_op_read.sum",
                    "lts__t_sectors_srcunit_tex_op_write.sum",
                    "lts__t_sectors_srcunit_tex.sum",
                    "lts__t_tag_requests.sum",
                    "l1tex__data_pipe_lsu_wavefronts.sum",
                    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum",
                    "l1tex__cycles_active.sum",
                ],
                codeText,
                "kernel",
                lc.block,
                lc.grid,
                lc.buffers,
                lc.alignmentBytes,
            )
            self.memLoad *= 1 / lc.lupCount
            self.memStore *= 1 / lc.lupCount
            self.L2Store *= 32 / lc.lupCount
            self.L2Load *= 32 / lc.lupCount
            self.L2Store_tex *= 32 / lc.lupCount
            self.L2Load_tex *= 32 / lc.lupCount
            self.L2Load = self.L2Load_tex
            self.L2Store = self.L2Store_tex
            self.L2tex *= 32 / lc.lupCount
            self.L2tagRequests *= 1 / lc.lupCount
            self.L1TagWavefronts *= 1 / lc.lupCount
            self.L1DataPipeWavefronts *= 1 / lc.lupCount
            self.L1CyclesActive *= 1 / lc.lupCount

        self.L2ltc = 1
        self.L2total = 1

        self.lups = lc.domain[0] * lc.domain[1] * lc.domain[2] / time / 1e9
        self.tflops = self.lups * lc.flops / 1000
        self.flopsPerLup = lc.flops

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

    def html(self, referenceCount="Lup"):
        return htmlColumnPrint(self, self.columns(), referenceCount)

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
