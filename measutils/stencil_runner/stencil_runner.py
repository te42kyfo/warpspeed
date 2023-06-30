#!/usr/bin/env python3

import os
from ctypes import *

filename = None
my_functions = None


def loadDLLs(API):
    global my_functions
    if my_functions is None:
        dirname = os.path.dirname(os.path.abspath(__file__))
        if len(dirname) == 0:
            dirname = "."
        if API == "HIP":
            filename = os.path.join(dirname, "hip_stencil_runner.so")
        elif API == "CUDA":
            filename = os.path.join(dirname, "cuda_stencil_runner.so")

        my_functions = CDLL(filename, mode=RTLD_GLOBAL)
        my_functions.timeKernel.restype = c_double
        my_functions.pyMeasureMetrics.restype = py_object
        my_functions.pyMeasureMetrics.argtypes = [
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

        my_functions.pyTimeNBufferKernel.restype = c_double


def time2BufferKernel(
    API, codeString, funcName, blockSize, gridSize, bufferSizeBytes, alignmentBytes
):
    loadDLLs(API)
    return c_double(
        my_functions.timeKernel(
            create_string_buffer(codeString.encode("utf-8")),
            create_string_buffer(funcName.encode("utf-8")),
            *blockSize,
            *gridSize,
            c_size_t(bufferSizeBytes),
            c_size_t(alignmentBytes)
        )
    ).value


def timeNBufferKernel(
    API, codeString, funcName, blockSize, gridSize, buffers, alignmentBytes
):
    loadDLLs(API)
    return c_double(
        my_functions.pyTimeNBufferKernel(
            codeString, funcName, blockSize, gridSize, buffers, alignmentBytes
        )
    ).value


def measureMetrics2BufferKernel(
    API,
    metricNames,
    codeString,
    funcName,
    blockSize,
    gridSize,
    bufferSizeBytes,
    alignmentBytes,
):
    loadDLLs(API)
    res = my_functions.pyMeasureMetrics(
        metricNames,
        codeString,
        funcName,
        blockSize,
        gridSize,
        bufferSizeBytes,
        alignmentBytes,
    )
    print("measureMetrics2BufferKernel: " + str(res))
    return res


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

        dt = timeNBufferKernel(
            "CUDA",
            codeString,
            "func",
            (256, 1, 1),
            (100000, 1, 1),
            [100000 * 256 * N * 8, 100000 * 256 * N * 9],
            0,
        )
        print(2 * 256 * 100000 * N * N / dt / 1.0e9)
