#!/usr/bin/env python3

from predict_metrics import *
from warpspeedkernel import *


blockSize = (256, 1, 2)
domain = (640, 512, 512)
blockingFactors = (1,1,1)
device = DeviceAmpere()

loads = [("tidx", "tidy", "tidz")]

for i in range(4):
    loads.extend([("tidx+" + str(i), "tidy", "tidz"),
                  ("tidx-" + str(i), "tidy", "tidz"),
                  ("tidx", "tidy+" + str(i), "tidz"),
                  ("tidx", "tidy-" + str(i), "tidz"),
                  ("tidx", "tidy", "tidz+" + str(i)),
                  ("tidx", "tidy", "tidz-" + str(i)),
                  ])

kernel =  WarpspeedGridKernel({"A" : loads},
                              {"B" : [("tidx", "tidy", "tidz")]},
                              domain, 32, 1)

lc = LaunchConfig.compute(kernel, blockSize, domain, blockingFactors, device)
basic = BasicMetrics.compute(lc, device, kernel)
pred = DerivedMetrics(lc, basic, device)


print(basic)
print(pred)
