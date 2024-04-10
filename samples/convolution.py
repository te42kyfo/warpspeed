#!/usr/bin/env python3


import sys

sys.path.append("../applications/unet/")
sys.path.append("../warpspeed")

import genconv
from predict_metrics import *
import sympy as sp
from warpspeedkernel import *

imageSize = (1024, 1024)
blockSize = (16, 1, 16)

device = DeviceAmpere()

input_channels = 64
output_channels = 64

for c_out_per_thread in [1]:
    for x_per_thread in [64]:
        print(str(x_per_thread) + " x " + str(c_out_per_thread))

        kernel = genconv.getConvWarpSpeedKernel(
            3,
            *imageSize,
            input_channels,
            output_channels,
            1,
            c_out_per_thread,
            x_per_thread,
            blockSize,
        )

        lc = LaunchConfig.compute(
            kernel,
            blockSize,
            (imageSize[0] - 2, imageSize[1] - 2, 1),
            (1, 1, 1),
            device,
            [],
        )
        basic = BasicMetrics.compute(lc, device, kernel)
        pred = DerivedMetrics(lc, basic, device)

        print(basic)
        print(pred)
