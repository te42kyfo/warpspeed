#!/usr/bin/env python3

from predict_metrics import *
import sympy as sp
from warpspeedkernel import *


blockSize = (1024, 1, 1)
domain = (1000000, 1, 1)
blockingFactors = (1, 1, 1)
device = DeviceAmpere()

# Each thread loads and stores a 3D vector. E.g. normalize and scale 3D vectors and store their lengths to another vector.

kernel = WarpspeedKernel(
    {"A": ["tidx*3 + 0", "tidx*3 + 1", "tidx*3 + 2"], "B": ["tidx"]},
    {"C": ["tidx*3 + 0", "tidx*3 + 1", "tidx*3 + 2"], "D": ["tidx"]},
    domain,
    32,
)

lc = LaunchConfig.compute(kernel, blockSize, domain, blockingFactors, device)
basic = BasicMetrics.compute(lc, device, kernel)
pred = DerivedMetrics(lc, basic, device)


print(basic)
print(pred)
