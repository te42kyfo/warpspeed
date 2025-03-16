#!/usr/bin/env python3

import sys

sys.path.append("../warpspeed/")

# from predict_metrics import *

from warpspeedkernel import *

arrayLength = 1024 * 1024 * 2
device = DeviceMI210()

kernel = WarpspeedKernel(
    loadFields=[
        Field(
            name="A",
            addresses=["tidx"],
            datatype=8,
            dimensions=(arrayLength, 1, 1),
            alignment=0,
        )
    ],
    storeFields=[
        Field(
            name="B",
            addresses=["tidx"],
            datatype=8,
            dimensions=(arrayLength, 1, 1),
            alignment=0,
        )
    ],
    registers=32,
    flops=1,
)

lc = LaunchConfig.compute(
    kernel,
    block=(1024, 1, 1),
    domain=(arrayLength, 1, 1),
    blocking_factors=(1, 1, 1),
    device=device,
)

print(lc)

basic = BasicMetrics.compute(lc, device, kernel, verbose=True)
pred = DerivedMetrics(lc, basic, device)

print(basic)
print(pred)
