#!/usr/bin/env python3


import sys

sys.path.append("../warpspeed/")

from predict_metrics import *
import sympy as sp
from warpspeedkernel import *


blockSize = (512, 1, 1)
domain = (15000, 15000, 1)
blockingFactors = (1, 1, 1)
device = DeviceAmpere()

kernel = WarpspeedKernel(
    loadFields=[
        Field(
            "A",
            [
                ("tidx", "tidy", "0"),
                ("tidx+1", "tidy", "0"),
                ("tidx-1", "tidy", "0"),
                ("tidx", "tidy+1", "0"),
                ("tidx", "tidy-1", "0"),
            ],
            8,
            domain,
            1,
        )
    ],
    storeFields=[Field("B", [("tidx", "tidy", "0")], 8, domain, 1)],
    registers=32,
    flops=4,
)

lc = LaunchConfig.compute(
    kernel, blockSize, domain, blockingFactors, device, (domain, domain)
)
basic = BasicMetrics.compute(lc, device, kernel)
pred = DerivedMetrics(lc, basic, device)


print(basic)
print(pred)
