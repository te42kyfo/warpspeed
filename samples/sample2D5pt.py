#!/usr/bin/env python3


import sys

sys.path.append("..")

from predict_metrics import *
import sympy as sp
from warpspeedkernel import *


blockSize = (512, 1, 1)
domain = (15000, 15000, 1)
blockingFactors = (1, 1, 1)
device = DeviceAmpere()

kernel = WarpspeedKernel(
    [
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
    [Field("B", [("tidx", "tidy", "0")], 8, domain, 1)],
    32
)

lc = LaunchConfig.compute(kernel, blockSize, domain, blockingFactors, device)
basic = BasicMetrics.compute(lc, device, kernel)
pred = DerivedMetrics(lc, basic, device)


print(basic)
print(pred)
