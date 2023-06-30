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
            filename = os.path.join(dirname, "hip_conv_runner.so")
        elif API == "CUDA":
            filename = os.path.join(dirname, "cuda_conv_runner.so")

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
        ]


def timeConvKernel(API, codeString, funcName, blockSize, gridSize, args):
    loadDLLs(API)
    return c_double(
        my_functions.timeKernel(
            create_string_buffer(codeString.encode("utf-8")),
            create_string_buffer(funcName.encode("utf-8")),
            *blockSize,
            *gridSize,
            *args
        )
    ).value


def measureMetricsConvKernel(
    API, metricNames, codeString, funcName, blockSize, gridSize, args
):
    loadDLLs(API)
    res = my_functions.pyMeasureMetrics(
        metricNames, codeString, funcName, blockSize, gridSize, args
    )
    return res


if __name__ == "__main__":
    input_channels = 120
    output_channels = 80
    width = 256
    height = 256
    batch_size = 1

    for c_in_per_thread in range(1, 20):
        for x_per_thread in range(1, 20):
            for xblock in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                for yblock in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                    for zblock in [1, 2, 4, 8, 16, 32, 64]:
                        if xblock * yblock * zblock != 256:
                            continue
                        if (
                            height % x_per_thread != 0
                            or input_channels % c_in_per_thread != 0
                        ):
                            continue

                        if (
                            width % xblock != 0
                            or height % (yblock * x_per_thread) != 0
                            or input_channels % (zblock * c_in_per_thread) != 0
                        ):
                            continue
                        codeString = """const int filter_size = {};
                                            const int c_in_per_thread = {};
                                            const int x_per_thread = {};""".format(
                            3, c_in_per_thread, x_per_thread
                        )

                        codeString += """

                extern "C" __global__ void __launch_bounds__(256)
                convolution_backward(float* error, float* next_error,
                                        float* weights, int batch_size, int input_channels,
                                        int output_channels, int width, int height) {

                int y = blockIdx.x * blockDim.x + threadIdx.x;
                int x0 = blockIdx.y * blockDim.y + threadIdx.y;
                int c_in0 = (blockIdx.z % input_channels) * blockDim.z + threadIdx.z;
                int b = blockIdx.z / input_channels;

                if (x0 * x_per_thread >= width || y >= height ||
                    c_in0 * c_in_per_thread >= input_channels)
                return;

                float val[c_in_per_thread][x_per_thread];
                for (int c = 0; c < c_in_per_thread; c++)
                for (int ix = 0; ix < x_per_thread; ix++)
                    val[c][ix] = 0.0f;

                for (int c_out = 0; c_out < output_channels; c_out++) {
                #pragma unroll
                for (int c = 0; c < c_in_per_thread; c++) {
                    int c_in = c_in0 * c_in_per_thread + c;
                #pragma unroll
                    for (int i = 0; i < filter_size; i++) {
                #pragma unroll
                    for (int j = 0; j < filter_size; j++) {
                        float w =
                            weights[c_out * input_channels * filter_size * filter_size +
                                    c_in * filter_size * filter_size +
                                    (filter_size - i - 1) * filter_size + filter_size - j - 1];
                #pragma unroll
                        for (int ix = 0; ix < x_per_thread; ix++) {
                        int x = x0 * x_per_thread + ix;
                        float error_val = error[b * output_channels * (width+1) * (height+1) +
                                                c_out * (width+1) * (height+1) +
                                                (x + i) * (height+1) + y + j];
                        val[c][ix] += error_val * w;
                        }
                    }
                    }
                }
                }
                for (int c = 0; c < c_in_per_thread; c++) {
                for (int ix = 0; ix < x_per_thread; ix++) {
                    int x = x0 * x_per_thread + ix;
                    int c_in = c_in0 * c_in_per_thread + c;
                    next_error[b * input_channels * (width+1) * (height + 1) +
                                c_in * (width+1) * (height + 1) +
                                x * (height+1) +
                                y] = val[c][ix];
                }
                }
                }
                    """

                        block_size = (xblock, yblock, zblock)

                        dt = timeConvKernel(
                            "CUDA",
                            codeString,
                            "convolution_backward",
                            block_size,
                            (
                                width // block_size[0],
                                height // block_size[1] // x_per_thread,
                                input_channels // block_size[2] // c_in_per_thread,
                            ),
                            [
                                input_channels,
                                output_channels,
                                batch_size,
                                width,
                                height,
                            ],
                        )

                        grid = (
                            width // block_size[0],
                            height // block_size[1] // x_per_thread,
                            input_channels // block_size[2] // c_in_per_thread,
                        )
                        metrics = measureMetricsConvKernel(
                            "CUDA",
                            [
                                "dram__bytes_read.sum",
                                "dram__bytes_write.sum",
                                "lts__t_sectors_srcunit_tex_op_read.sum",
                                "lts__t_sectors_srcunit_tex_op_write.sum",
                            ],
                            codeString,
                            "convolution_backward",
                            block_size,
                            grid,
                            (
                                input_channels,
                                output_channels,
                                batch_size,
                                width,
                                height,
                            ),
                        )

                        flops = (
                            9 * 2 * output_channels * input_channels * width * height
                        )
                        threads = xblock * yblock * zblock * grid[0] * grid[1] * grid[2]
                        print(
                            "({:3}, {:3}, {:2}) {:4}  {:4}  {:5.2f} {:6.0f}: {:6.0f}  | {:6.3f} {:6.3f} {:6.3f} {:6.3f}".format(
                                xblock,
                                yblock,
                                zblock,
                                c_in_per_thread,
                                x_per_thread,
                                dt * 1000,
                                flops,
                                flops / dt / 1e9,
                                metrics[0] / flops,
                                metrics[1] / flops,
                                metrics[2] * 32 / flops,
                                metrics[3] * 32 / flops,
                            )
                        )
