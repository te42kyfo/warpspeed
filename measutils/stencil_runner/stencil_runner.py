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


def time2BufferKernel(API, codeString, funcName, blockSize, gridSize, bufferSizeBytes, alignmentBytes):
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


def measureMetrics2BufferKernel(
        API, metricNames, codeString, funcName, blockSize, gridSize, bufferSizeBytes, alignmentBytes
):
    loadDLLs(API)
    res = my_functions.pyMeasureMetrics(
        metricNames, codeString, funcName, blockSize, gridSize, bufferSizeBytes, alignmentBytes
    )
    print("measureMetrics2BufferKernel: " + str(res))
    return res;

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

        codeString = """
        #define int64_t int
        extern "C" __global__  __launch_bounds__(512) void func(double * __restrict__ _data_dst, double * __restrict__ const _data_src)
{
   if (blockDim.x*blockIdx.x + threadIdx.x + 4 < 636 && blockDim.y*blockIdx.y + threadIdx.y + 4 < 508 && 2*blockDim.z*blockIdx
.z + 2*threadIdx.z + 4 < 996)
   {
      const int64_t ctr_0 = blockDim.x*blockIdx.x + threadIdx.x + 4;
      const int64_t ctr_1 = blockDim.y*blockIdx.y + threadIdx.y + 4;
      const int64_t ctr_2 = 2*blockDim.z*blockIdx.z + 2*threadIdx.z + 4;
      double * __restrict__ _data_src_10_2m2 = _data_src + 640*ctr_1 + 327680*ctr_2 - 655360;
      const double fa_9 = _data_src_10_2m2[ctr_0];
      double * __restrict__ _data_src_10_2m1 = _data_src + 640*ctr_1 + 327680*ctr_2 - 327680;
      const double fa_0 = _data_src_10_2m1[ctr_0];
      double * __restrict__ _data_src_1m2_20 = _data_src + 640*ctr_1 + 327680*ctr_2 - 1280;
      const double fa_8 = _data_src_1m2_20[ctr_0];
      double * __restrict__ _data_src_1m1_20 = _data_src + 640*ctr_1 + 327680*ctr_2 - 640;
      const double fa_10 = _data_src_1m1_20[ctr_0];
      double * __restrict__ _data_src_10_20 = _data_src + 640*ctr_1 + 327680*ctr_2;
      const double fa_6 = _data_src_10_20[ctr_0 - 2];
      const double fa_7 = _data_src_10_20[ctr_0 - 1];
      const double fa_3 = _data_src_10_20[ctr_0];
      const double fa_12 = _data_src_10_20[ctr_0 + 1];
      const double fa_11 = _data_src_10_20[ctr_0 + 2];
      double * __restrict__ _data_src_11_20 = _data_src + 640*ctr_1 + 327680*ctr_2 + 640;
      const double fa_5 = _data_src_11_20[ctr_0];
      double * __restrict__ _data_src_12_20 = _data_src + 640*ctr_1 + 327680*ctr_2 + 1280;
      const double fa_4 = _data_src_12_20[ctr_0];
      double * __restrict__ _data_src_1m2_21 = _data_src + 640*ctr_1 + 327680*ctr_2 + 326400;
      const double fa_15 = _data_src_1m2_21[ctr_0];
      double * __restrict__ _data_src_1m1_21 = _data_src + 640*ctr_1 + 327680*ctr_2 + 327040;
      const double fa_21 = _data_src_1m1_21[ctr_0];
      double * __restrict__ _data_src_10_21 = _data_src + 640*ctr_1 + 327680*ctr_2 + 327680;
      const double fa_13 = _data_src_10_21[ctr_0 - 2];
      const double fa_18 = _data_src_10_21[ctr_0 - 1];
      const double fa_2 = _data_src_10_21[ctr_0];
      const double fa_20 = _data_src_10_21[ctr_0 + 1];
      const double fa_14 = _data_src_10_21[ctr_0 + 2];
      double * __restrict__ _data_src_11_21 = _data_src + 640*ctr_1 + 327680*ctr_2 + 328320;
      const double fa_17 = _data_src_11_21[ctr_0];
      double * __restrict__ _data_src_12_21 = _data_src + 640*ctr_1 + 327680*ctr_2 + 328960;
      const double fa_16 = _data_src_12_21[ctr_0];
      double * __restrict__ _data_src_10_22 = _data_src + 640*ctr_1 + 327680*ctr_2 + 655360;
      const double fa_1 = _data_src_10_22[ctr_0];
      double * __restrict__ _data_src_10_23 = _data_src + 640*ctr_1 + 327680*ctr_2 + 983040;
      const double fa_19 = _data_src_10_23[ctr_0];
      const double xi_0 = fa_0*0.25 + fa_1*0.25 + fa_2*0.25 + fa_3*0.25;
      double * __restrict__ _data_dst_10_20 = _data_dst + 640*ctr_1 + 327680*ctr_2;
      _data_dst_10_20[ctr_0] = fa_10*0.25 + fa_11*0.25 + fa_12*0.25 + fa_4*0.25 + fa_5*0.25 + fa_6*0.25 + fa_7*0.25 + fa_8*0.25 + fa_9*0.25 + xi_0;
      double * __restrict__ _data_dst_10_21 = _data_dst + 640*ctr_1 + 327680*ctr_2 + 327680;
      _data_dst_10_21[ctr_0] = fa_13*0.25 + fa_14*0.25 + fa_15*0.25 + fa_16*0.25 + fa_17*0.25 + fa_18*0.25 + fa_19*0.25 + fa_20*0.25 + fa_21*0.25 + xi_0;
   } 
}"""

        dt = time2BufferKernel(
            "CUDA",
            codeString,
            "func",
            (256, 1, 1),
            (100000, 1, 1),
            100000 * 256 * N * 8,
        )
        print(2 * 256 * 100000 * N * N / dt / 1.0e9)
