#!/usr/bin/env python3


import sys

sys.path.append("../../warpspeed/")

from predict_metrics import *
import sympy as sp
from warpspeedkernel import *

imageSize = (1024, 1024)
blockSize = (16, 16, 1)

device = DeviceAmpere()

c_out = 120
c_in = 80

for x_per_thread in [1, 4, 8]:
    for c_in_per_thread in [1, 4, 8]:
        print(str(x_per_thread) + " x " + str(c_in_per_thread))

        loadFields = []
        storeFields = []

        errorLoads = []

        for iy in range(x_per_thread):
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    errorLoads.append(
                        (
                            "tidx + " + str(i),
                            "tidy * {} + {} + {}".format(x_per_thread, iy, j),
                            "0",
                        )
                    )

        print(errorLoads)

        loadFields.append(
            Field(
                name="errors",
                addresses=errorLoads,
                datatype=4,
                dimensions=imageSize,
                alignment=0,
                multiplicity=c_out,
            )
        )

        weightLoads = []
        for ic in range(c_in_per_thread):
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    weightLoads.append(
                        (
                            str(i),
                            str(j),
                            "tidz * " + str(c_in_per_thread) + " + " + str(ic),
                        )
                    )

        loadFields.append(
            Field(
                name="weights",
                addresses=weightLoads,
                datatype=4,
                dimensions=(*imageSize, c_in),
                alignment=0,
                multiplicity=c_out,
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
                dimensions=(*imageSize, c_in),
                alignment=0,
                multiplicity=1,
            )
        )

        kernel = WarpspeedKernel(
            loadFields,
            storeFields,
            registers=128,
            flops=c_out * c_in_per_thread * x_per_thread * 3 * 3 * 2,
        )

        grid = (
            int(ceil(imageSize[0] / blockSize[0])),
            int(ceil(imageSize[1] / blockSize[1])) // x_per_thread,
            int(ceil(c_in / blockSize[2])) // c_in_per_thread,
        )

        lc = LaunchConfig.compute(kernel, blockSize, grid, (1, 1, 1), device, 12123)

        basic = BasicMetrics.compute(lc, device, kernel)
        print(basic)

        pred = DerivedMetrics(lc, basic, device)

        print(pred)
