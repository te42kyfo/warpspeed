#!/usr/bin/env python3
import sys
sys.path.append('..')


from predict_metrics import *
import sympy as sp
from warpspeedkernel import *

import matplotlib.pyplot as plt


blockSize = (512, 1, 1)
domain = (1024, 1024, 256)
blockingFactors = (1, 1, 1)


device = DeviceAmpere()


xySizes = [1, 2, 4, 8, 16] + [i * 32 for i in range(1, 16)] + [1024]
preds = []

for xy in xySizes:
    domain = (xy, xy, 512 * 256 * 256 // (xy * xy))

    loadFields = []
    storeFields = []

    xloads = []
    for z in {-1, 0, 1}:
        for y in {-1, 0, 1}:
            for x in {-1, 0, 1}:
                xloads.append(
                    ("tidx + " + str(x), "tidy + " + str(y), "tidz + " + str(z))
                )
    loadFields.append(Field("X", xloads, 8, [d + 2 for d in domain], 0))

    matrixLoads = []
    for row in range(0, 27):
        matrixLoads.append(
            (
                "(tidx + tidy * {0} + tidz * {0} * {1}) + {3} * {0} * {1} * {2}".format(
                    domain[0], domain[1], domain[2], row
                ),
                "0",
                "0",
            )
        )

    loadFields.append(
        Field("mat", matrixLoads, 8, (domain[0], domain[1], domain[2] * 27), 0)
    )
    loadFields.append(
        Field("idx", matrixLoads, 4, (domain[0], domain[1], domain[2] * 27), 0)
    )
    storeFields.append(
        Field("Y", [("tidx", "tidy", "tidz")], 8, [d + 2 for d in domain], 0)
    )

    kernel = WarpspeedKernel(loadFields, storeFields, 64, flops=27 * 2)

    lc = LaunchConfig.compute(kernel, blockSize, domain, blockingFactors, device)
    basic = BasicMetrics.compute(lc, device, kernel)
    pred = DerivedMetrics(lc, basic, device)

    preds.append(pred)

    print(basic)
    print(pred)


fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(xySizes, [pred.perfV3 for pred in preds])
ax.set_ylim((0, ax.get_ylim()[1]))

plt.show()
