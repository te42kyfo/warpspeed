#!/usr/bin/env python3

import sys

sys.path.append("../warpspeed/")

from predict_metrics import *
from warpspeedkernel import *


def getCodeString(
    filter_size,
    width,
    height,
    input_channels,
    output_channels,
    batch_size,
    c_in_per_thread,
    x_per_thread,
    block_size,
):
    codeString = """
const int filter_size = {};
const int c_in_per_thread = {};
const int x_per_thread = {};
const int input_channels = {};
const int output_channels = {};
const int blockx = {};
const int blocky = {};
const int blockz = {};
const int width = {};
const int height = {};
//const bool useZero = true;
    """.format(
        filter_size,
        c_in_per_thread,
        x_per_thread,
        input_channels,
        output_channels,
        *block_size,
        height,
        width,  # width and height are swapped!
    )
    codeString += """
extern "C" __global__ void __launch_bounds__(256)
kernel(float* error, float* next_error,
                        float* weights) {
    int zero = 1;

    int y = blockIdx.x * blockx * zero + threadIdx.x;
    int x0 = blockIdx.y * blocky * zero + threadIdx.y;
    int c_in0 = blockIdx.z * blockz * zero;

    if(blockz > 1) {
        c_in0 += threadIdx.z;
        if (blockx*blocky >= 64) {
            #ifdef __HIPCC__
                asm("v_readfirstlane_b32 %0, %1" : "=s"(c_in0) : "v"(c_in0));
            #endif
        }
    }


    if (x0 * x_per_thread >= width || y >= height || c_in0 * c_in_per_thread >= input_channels)
        return;

    float val[c_in_per_thread][x_per_thread];
    for (int c = 0; c < c_in_per_thread; c++) {
        for (int ix = 0; ix < x_per_thread; ix++) {
            val[c][ix] = 0.0f;
        }
    }


    for (int c_out = 0; c_out < output_channels; c_out++) {

        #pragma unroll
        for (int c = 0; c < c_in_per_thread; c++) {
            int c_in = c_in0 * c_in_per_thread + c;

            #pragma unroll
            for (int i = 0; i < filter_size; i++) {

                #pragma unroll
                for (int j = 0; j < filter_size; j++) {
                    float w = weights[zero * c_out * input_channels * filter_size * filter_size +
                                (c_in+0) * filter_size * filter_size + i * filter_size + j];

                #pragma unroll
                    for (int ix = 0; ix < x_per_thread; ix++) {
                        int x = x0 * x_per_thread + ix;
                        float error_val = error[zero * c_out * (width+2) * (height+2) +
                                                (x + i) * (height+2) + y + j];
                        val[c][ix] = val[c][ix] + error_val * w;

                    }
                }
            }
        }
    }
    for (int c = 0; c < c_in_per_thread; c++) {
        for (int ix = 0; ix < x_per_thread; ix++) {
            int x = x0 * x_per_thread + ix;
            int c_in = c_in0 * c_in_per_thread + c;

            next_error[(c_in+0) * (width+2) * (height + 2) + x * (height+2) + y] = val[c][ix];

            }
    }

}
                    """

    domain = [
        width,
        height // x_per_thread,
        input_channels // c_in_per_thread,
    ]

    buffers = [
        (width + 1) * (height + 1) * batch_size * output_channels * 4,
        (width + 1) * (height + 1) * batch_size * input_channels * 4,
        input_channels * output_channels * 3 * 3 * 4,
    ]
    return codeString, buffers, domain


def getConvWarpSpeedKernel(
    filterSize,
    width,
    height,
    input_channels,
    output_channels,
    batch_size,
    c_in_per_thread,
    x_per_thread,
    blockSize,
):
    loadFields = []
    storeFields = []

    errorLoads = []

    for iy in range(x_per_thread):
        for j in [0, 1, 2]:
            for i in [0, 1, 2]:
                errorLoads.append(
                    (
                        "tidx + " + str(i),
                        "tidy * {} + {} + {}".format(x_per_thread, iy, j),
                        "0",
                    )
                )

    loadFields.append(
        Field(
            name="errors",
            addresses=errorLoads,
            datatype=4,
            dimensions=(width + 2, height + 2),
            alignment=0,
            multiplicity=output_channels,
        )
    )
    # for a in errorLoads:
    #    print(a)

    weightLoads = []
    for ic in range(c_in_per_thread):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                weightLoads.append(
                    (
                        "{} + {} * 3 + (tidz* {} + {}) * 3 * 3".format(
                            i, j, c_in_per_thread, ic
                        ),
                        0,
                        0,
                    ),
                )

    loadFields.append(
        Field(
            name="weights",
            addresses=weightLoads,
            datatype=4,
            dimensions=(3 * 3 * input_channels, 1, 1),
            alignment=0,
            multiplicity=output_channels,
            scalar=True if blockSize[1] * blockSize[0] >= 64 else False,
        )
    )

    outputStores = []
    for ic in range(c_in_per_thread):
        for iy in range(x_per_thread):
            outputStores.append(
                (
                    "tidx",
                    "tidy * {} + {}".format(x_per_thread, iy),
                    "tidz * {} + {}".format(c_in_per_thread, ic),
                )
            )

    storeFields.append(
        Field(
            name="output",
            addresses=outputStores,
            datatype=4,
            dimensions=(width + 2, height + 2, input_channels),
            alignment=0,
            multiplicity=1,
        )
    )

    kernel = WarpspeedKernel(
        loadFields,
        storeFields,
        registers=128,
        flops=output_channels * c_in_per_thread * x_per_thread * 3 * 3 * 2,
    )

    return kernel

    # grid = (
    #    int(ceil(imageSize[0] / blockSize[0])),
    #    int(ceil(imageSize[1] / blockSize[1])) // x_per_thread,
    #    int(ceil(c_in / blockSize[2])) // c_in_per_thread,
    # )

    # lc = LaunchConfig.compute(kernel, blockSize, grid, (1, 1, 1), device, 12123)

    # basic = BasicMetrics.compute(lc, device, kernel)

    # pred = DerivedMetrics(lc, basic, device)
