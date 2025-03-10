#!/usr/bin/env python3


import sys

sys.path.append("../warpspeed/")

from predict_metrics import *
from warpspeedkernel import *


device = DeviceAmpereA100_80GB()

kernel = WarpspeedKernel(
    loadFields=[
        Field(
            name="A",
            addresses=[
                ("tidx", "tidy", "0"),
                ("tidx+1", "tidy", "0"),
                ("tidx-1", "tidy", "0"),
                ("tidx", "tidy+1", "0"),
                ("tidx", "tidy-1", "0"),
            ],
            datatype=8,
            dimensions=(15002, 15002, 1),
            alignment=1,
        )
    ],
    storeFields=[Field("B", [("tidx", "tidy", "0")], 8, (15002, 15002, 1), 1)],
    registers=32,
    flops=5,
)

lc = LaunchConfig.compute(
    kernel=kernel,
    block=(512, 1, 1),
    domain=(15000, 15000, 1, 1),
    blocking_factors=(1, 1, 1),
    device=device,
)
basic = BasicMetrics.compute(lc, device, kernel)
pred = DerivedMetrics(lc, basic, device)

print(lc)
print(basic)
print(pred)
