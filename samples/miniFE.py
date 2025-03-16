#!/usr/bin/env python3
import sys

sys.path.append("../warpspeed")


from predict_metrics import *
import sympy as sp
from warpspeedkernel import *


blockSize = (512, 1, 1)
domain = (1024, 1024, 256)
blockingFactors = (1, 1, 1)


device = DeviceAmpereA100_80GB()


xySizes = [64, 256, 1024]
preds = []


for vectorCount in [1, 4, 16]:
    for xy in [64, 256, 1024]:
        domain = (xy, xy, 512 * 256 * 256 // (xy * xy))

        print("Domain: " + str(domain) + ", " + str(vectorCount) + " vectors")

        loadFields = []
        storeFields = []

        xloads = []
        for z in {-1, 0, 1}:
            for y in {-1, 0, 1}:
                for x in {-1, 0, 1}:
                    xloads.append(
                        ("tidx + " + str(x), "tidy + " + str(y), "tidz + " + str(z))
                    )
        loadFields.append(
            Field("X", xloads, 8, [d + 2 for d in domain], 0, vectorCount)
        )

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
            Field(
                "Y",
                [("tidx", "tidy", "tidz")],
                8,
                [d + 2 for d in domain],
                0,
                vectorCount,
            )
        )

        kernel = WarpspeedKernel(
            loadFields, storeFields, 64, flops=27 * 2 * vectorCount
        )

        lc = LaunchConfig.compute(
            kernel, blockSize, domain, blockingFactors, device, []
        )
        basic = BasicMetrics.compute(lc, device, kernel)
        pred = DerivedMetrics(lc, basic, device)

        preds.append(pred)

        print(basic)
        print(pred)
