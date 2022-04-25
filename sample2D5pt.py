#!/usr/bin/env python3

from predict_metrics import *
import sympy as sp
from warpspeedkernel import *


blockSize = (256, 2, 1)
domain = (15000, 15000, 1)
blockingFactors = (1,1,1)
device = DeviceAmpere()

kernel =  WarpspeedGridKernel({"A" : [("tidx+1", "tidy", "0"),
                                      ("tidx-1", "tidy", "0"),
                                      ("tidx", "tidy+1", "0"),
                                      ("tidx", "tidy-1", "0")]},
                              {"B" : [("tidx", "tidy", "0")]},
                              domain, 32, 1)

lc = LaunchConfig.compute(kernel, blockSize, domain, blockingFactors, device)
basic = BasicMetrics.compute(lc, device, kernel)
pred = DerivedMetrics(lc, basic, device)


print(basic)
print(pred)
