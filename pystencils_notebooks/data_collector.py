#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.append("../warpspeed/")
sys.path.append("../measutils/")
sys.path.append("../measutils/stencil_runner")
sys.path.append("../pystencils")
sys.path.append("../../pystencils")

sys.path.append("..")


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from meas_db import MeasDB

import pystencils as ps
from pystencils.slicing import add_ghost_layers, make_slice, remove_ghost_layers
from pystencils.display_utils import show_code, get_code_str

import sympy as sp

from measured_metrics import MeasuredMetrics, ResultComparer
import stencil_runner
from predict_metrics import *
from plot_utils import *
import json
from datetime import datetime


from pystencils_stencil_utils import getStencilKernel
from pystencilswarpspeedkernel import PyStencilsWarpSpeedKernel

meas_db = MeasDB("stencils.db")
# meas_db.clearDB()

API = stencil_runner.getAPI()
print("The API is: ", API)

import socket

print("We are running on: ", socket.gethostname())


# In[5]:

fieldSize = (1026, 512, 200)
datatype = "double"


def getBlockSizes(threadCounts):
    blockSizes = []
    for xblock in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for yblock in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            for zblock in [1, 2, 4, 8, 16, 32, 64]:
                if xblock * yblock * zblock in threadCounts:
                    blockSizes.append((xblock, yblock, zblock))
    return blockSizes


remeasure = False

for deviceId in range(0, 8):
    driverDeviceName = stencil_runner.getDeviceName(deviceId)
    print("deviceID", deviceId, ", the driver calls this: ", driverDeviceName)
    device = selectDevice(driverDeviceName)

    if device is None:
        print("No device found")
        continue
    print(
        "The detected device's driver name is",
        device.name,
        "and the display name is",
        device.displayName,
    )

    for r in [-2, -1, 0, 1, 2, 3, 4]:
        for blockingFactors in [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2)]:
            for blockSize in getBlockSizes([1024]):

                t1 = time.process_time()

                key = (r, *blockSize, *blockingFactors)

                date, lc, basic, meas = meas_db.getEntry(
                    r, blockSize, blockingFactors, fieldSize, datatype, device
                )

                if not meas is None and remeasure is False:
                    print(key, " found in cache (", date, "), skip")
                else:

                    if meas is None:
                        print(key, " not found in cache, measure")
                    else:
                        print(key, " found in cache (", date, "), replace")

                    if r >= 0:
                        kernel, domainSize, bufferSizeBytes, alignmentBytes = (
                            getStencilKernel(
                                r,
                                "star",
                                blockSize,
                                blockingFactors,
                                fieldSize,
                                datatype,
                            )
                        )
                    else:
                        kernel, domainSize, bufferSizeBytes, alignmentBytes = (
                            getStencilKernel(
                                -r,
                                "box",
                                blockSize,
                                blockingFactors,
                                fieldSize,
                                datatype,
                            )
                        )

                    wsKernel = PyStencilsWarpSpeedKernel(kernel)
                    wsKernel.registers = 32

                    buffers = [bufferSizeBytes, bufferSizeBytes]
                    lc = LaunchConfig.compute(
                        wsKernel,
                        blockSize,
                        domainSize,
                        blockingFactors,
                        device,
                        buffers,
                        alignmentBytes,
                    )

                    basic = BasicMetrics.compute(lc, device, wsKernel)
                    newMeas = MeasuredMetrics.measure(get_code_str(kernel), lc)

                    meas_db.insertValue(
                        r,
                        blockSize,
                        blockingFactors,
                        device,
                        basic,
                        newMeas,
                        lc,
                        fieldSize,
                        datatype,
                    )

                    print(newMeas)

                    print(
                        "{:.1f} TFlop/s, {:.1f} GB/s".format(
                            newMeas.tflops,
                            newMeas.lups * 2 * (8 if datatype == "double" else 4),
                        )
                    )
                    if not meas is None:
                        print(
                            "Time diff: {:.2f}%".format(
                                (meas.time - newMeas.time) / meas.time * 100
                            ),
                        )

                meas_db.commit()
                t2 = time.process_time()
