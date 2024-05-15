#!/usr/bin/env python3

import os
from ctypes import *
import socket

filename = None
my_functions = None


def loadDLLs(API):
    global my_functions
    if my_functions is None:
        dirname = os.path.dirname(os.path.abspath(__file__))
        if len(dirname) == 0:
            dirname = "."
        if API == "HIP":
            filename = os.path.join(
                dirname, "hip_stencil_runner." + socket.gethostname() + ".so"
            )
        elif API == "CUDA":
            filename = os.path.join(
                dirname, "cuda_stencil_runner." + socket.gethostname() + ".so"
            )

        my_functions = CDLL(filename, mode=RTLD_GLOBAL)
        my_functions.pyMeasureMetricsNBufferKernel.restype = py_object
        my_functions.pyMeasureMetricsNBufferKernel.argtypes = [
            py_object,
            py_object,
            py_object,
            py_object,
            py_object,
            py_object,
            py_object,
        ]

        my_functions.pyTimeNBufferKernel.argtypes = [
            py_object,
            py_object,
            py_object,
            py_object,
            py_object,
            py_object,
        ]

        my_functions.pyTimeNBufferKernel.restype = py_object


def timeNBufferKernel(
    API, codeString, funcName, blockSize, gridSize, buffers, alignmentBytes
):
    loadDLLs(API)
    return my_functions.pyTimeNBufferKernel(
        codeString, funcName, blockSize, gridSize, buffers, alignmentBytes
    )


def measureMetricsNBufferKernel(
    API,
    metricNames,
    codeString,
    funcName,
    blockSize,
    gridSize,
    buffers,
    alignmentBytes,
):
    loadDLLs(API)
    return my_functions.pyMeasureMetricsNBufferKernel(
        metricNames,
        codeString,
        funcName,
        blockSize,
        gridSize,
        buffers,
        alignmentBytes,
    )


if __name__ == "__main__":
    for N in range(1, 190):
        codeString = 'extern "C" __global__ void func(double* A, double* B) {\n'
        codeString += (
            "   int tidx = threadIdx.x + blockIdx.x*blockDim.x;\n double prod = 0.0;\n"
        )

        for i in range(0, N):
            for n in range(0, N):
                codeString += (
                    "prod += "
                    + "A[(size_t) tidx*{0}+{1}] * B[(size_t) tidx*{0} + {2}];\n".format(
                        N, i, n
                    )
                )

        codeString += "if (prod == 12.3213) A[0] = prod;}"

        results = timeNBufferKernel(
            "CUDA",
            codeString,
            "func",
            (256, 1, 1),
            (100000, 1, 1),
            [100000 * 256 * N * 8, 100000 * 256 * N * 9],
            0,
        )
        print(2 * 256 * 100000 * N * N / results["time"] / 1.0e9)

        values = measureMetricsNBufferKernel(
            "CUDA",
            ["dram__bytes_read.sum"],
            codeString,
            "func",
            (256, 1, 1),
            (100000, 1, 1),
            [100000 * 256 * N * 8, 100000 * 256 * N * 9],
            0,
        )
        print(values[0] / 256 / 100000)
