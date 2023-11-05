#!/usr/bin/env python3

import sys

sys.path.append("../warpspeed/")

from predict_metrics import *
from warpspeedkernel import *


blockSize = (256, 1, 2)
domain = (640, 512, 512)
blockingFactors = (1, 1, 1)
device = DeviceAmpere()

loads = [("tidx", "tidy", "tidz")]

stencilRange = 4

for i in range(1, stencilRange + 1):
    loads.extend(
        [
            ("tidx+" + str(i), "tidy", "tidz"),
            ("tidx-" + str(i), "tidy", "tidz"),
            ("tidx", "tidy+" + str(i), "tidz"),
            ("tidx", "tidy-" + str(i), "tidz"),
            ("tidx", "tidy", "tidz+" + str(i)),
            ("tidx", "tidy", "tidz-" + str(i)),
        ]
    )


kernel = WarpspeedKernel(
    loadFields=[Field("A", loads, 8, domain, stencilRange)],
    storeFields=[Field("B", [("tidx", "tidy", "tidz")], 8, domain, stencilRange)],
    registers=32,
    flops=25,
)


lc = LaunchConfig.compute(kernel, blockSize, domain, blockingFactors, device, [domain])
basic = BasicMetrics.compute(lc, device, kernel)
pred = DerivedMetrics(lc, basic, device)


print(basic)
print(pred)
