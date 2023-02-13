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
    [Field("A", loads, 8, domain, stencilRange)],
    [Field("B", [("tidx", "tidy", "tidz")], 8, domain, stencilRange)],
    32,
)


lc = LaunchConfig.compute(
    kernel, blockSize, domain, blockingFactors, device
)
basic = BasicMetrics.compute(lc, device, kernel)
pred = DerivedMetrics(lc, basic, device)


print(basic)
print(pred)
