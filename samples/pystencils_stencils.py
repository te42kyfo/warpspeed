#!/usr/bin/env python3

import sys

sys.path.append("../../pystencils")
sys.path.append("..")
sys.path.append("../pystencils")


from pystencilswarpspeedkernel import PyStencilsWarpSpeedKernel
import pystencils as ps
from pystencils_stencil_utils import PS3DStencil
from warpspeedkernel import *


size = (640, 512, 1000)
device = DeviceAmpere()
stencilRange = 1

SS = PS3DStencil(size, stencilRange)



for stencilRange, threadBlockSize, threadFolding in [(0, (512, 1, 1), (1,1,1)),
                                                     (0, (1, 1, 64), (1, 1, 1)),
                                                     (1, (512, 1, 1), (1, 1, 1)),
                                                     (1, (32, 1, 32), (1, 1, 1)),
                                                     (1, (32, 1, 32), (1, 2, 1)),
                                                     (1, (1, 1, 64), (1, 1, 1)),
                                                     (4, (512, 1, 1), (1, 1, 1)),
                                                     (4, (1, 1, 64), (1, 1, 1)),
                                                     (4, (64, 4, 1), (1, 1, 2))]:


    print("{} {} {}".format(stencilRange, threadBlockSize, threadFolding))
    kernel = SS.getStarKernel(threadBlockSize, stencilRange, threadFolding)

    wsKernel = PyStencilsWarpSpeedKernel(kernel.ast)

    lc = LaunchConfig.compute(
        wsKernel, threadBlockSize, [d - stencilRange for d in size], threadFolding, device)
    basic = BasicMetrics.compute(lc, device, wsKernel)
    pred = DerivedMetrics(lc, basic, device)

    print(pred)
