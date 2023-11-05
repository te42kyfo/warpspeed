#!/usr/bin/env python3

import sys

sys.path.append("../warpspeed/")

from predict_metrics import *
import sympy as sp
from warpspeedkernel import *


blockSize = (1024, 1, 1)
domain = (1000000, 1, 1)
blockingFactors = (1, 1, 1)
device = DeviceAmpere()

# Each thread loads and stores a 3D vector. E.g. normalize and scale 3D vectors and store their lengths to another vector.


kernel = WarpspeedKernel(
    loadFields=[
        Field("A", ["tidx*3 + 0", "tidx*3 + 1", "tidx*3 + 2"], 8, domain, 0),
        Field("B", ["tidx"], 8, domain, 0),
    ],
    storeFields=[
        Field("D", ["tidx*3 + 0", "tidx*3 + 1", "tidx*3 + 2"], 8, domain, 0),
        Field("B", ["tidx"], 8, domain, 0),
    ],
    registers=32,
    flops=13,
)

buffers = [domain, domain, domain, domain]

lc = LaunchConfig.compute(kernel, blockSize, domain, blockingFactors, device, buffers)
basic = BasicMetrics.compute(lc, device, kernel)
pred = DerivedMetrics(lc, basic, device)


print(basic)
print(pred)
