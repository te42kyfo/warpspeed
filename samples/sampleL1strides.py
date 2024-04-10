#!/usr/bin/env python3


#!/usr/bin/env python3

import sys

sys.path.append("../warpspeed/")

from predict_metrics import *
import sympy as sp
from warpspeedkernel import *


blockingFactors = (1, 1, 1)
device = DeviceAmpereA100_80GB()

for datatype in [4, 8]:

    for blockx in [1, 2, 4, 8, 16, 32, 64]:
        blockSize = (blockx, 1024 // blockx, 1)
        for pitch in [2]:
            domain = (1024 + pitch, 1024, 1)

            kernel = WarpspeedKernel(
                loadFields=[
                    Field("A", [("tidx", "tidy", "tidz")], datatype, domain, 0)
                ],
                storeFields=[],
                registers=32,
                flops=13,
            )

            buffers = [domain, domain, domain, domain]

            lc = LaunchConfig.compute(
                kernel, blockSize, domain, blockingFactors, device, buffers
            )

            basic = BasicMetrics.compute(lc, device, kernel)
            # pred = DerivedMetrics(lc, basic, device)

            print(
                "double" if datatype == 8 else "float",
                "block",
                blockx,
                pitch,
                basic.L1Cycles * 32,
                datatype / basic.L1Cycles,
                basic.L1DataPipeCycles * 32,
                basic.L1TagCycles * 32,
            )

    for stride in range(33):
        blockSize = (1024, 1, 1)

        domain = (1024 * 1024, 1, 1)

        kernel = WarpspeedKernel(
            loadFields=[
                Field(
                    "A",
                    [("tidx * " + str(stride), "tidy", "tidz")],
                    datatype,
                    domain,
                    0,
                )
            ],
            storeFields=[],
            registers=32,
            flops=13,
        )

        buffers = [domain, domain, domain, domain]

        lc = LaunchConfig.compute(
            kernel, blockSize, domain, blockingFactors, device, buffers
        )

        basic = BasicMetrics.compute(lc, device, kernel)
        # pred = DerivedMetrics(lc, basic, device)

        print(
            "double" if datatype == 8 else "float",
            "stride",
            stride,
            stride,
            basic.L1Cycles * 32,
            datatype / basic.L1Cycles,
            basic.L1DataPipeCycles * 32,
            basic.L1TagCycles * 32,
        )
    print()
