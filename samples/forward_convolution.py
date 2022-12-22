#!/usr/bin/env python3

import sys
sys.path.append('..')

from predict_metrics import *
import sympy as sp
from warpspeedkernel import *

imageSize = (1024, 1024)
blockSize = (512, 1, 1)

device = DeviceAmpere()

for imageCount in [1, 8]:
    for inputChannels in [1, 4, 8]:
        for outputChannels in [1,4,8]:
            print(str(inputChannels) + " x " + str(outputChannels))

            loadFields = []
            storeFields = []

            imageLoads = []

            for imageChannel in range(inputChannels):
                for x in [-1,0,1]:
                    for y in [-1, 0, 1]:
                        imageLoads.append(("tidx + " + str(x), "tidy + " + str(y), str(imageChannel) ))

            stencilLoads = []
            for x in range(3):
                for y in range(3):
                    for outputChannel in range(outputChannels):
                        stencilLoads.append((  str(x), str(y), str(outputChannel) ))

            loadFields.append(Field(name = "stencil",
                                    addresses = stencilLoads,
                                    datatype = 4,
                                    dimensions = (3,3,outputChannels),
                                    alignment = 0))

            for image in range(imageCount):
                loadFields.append(Field(name = "images",
                                    addresses = imageLoads,
                                    datatype = 4,
                                    dimensions = (*imageSize, inputChannels),
                                    alignment = 0 ))


            outputStores = []
            for outputChannel in range(outputChannels):
                outputStores.append(("tidx", "tidy", str(outputChannel) ))

            for image in range(imageCount):
                storeFields.append(Field(name = "output",
                                        addresses = outputStores,
                                        datatype = 4,
                                        dimensions = (*imageSize, outputChannels),
                                        alignment = 0))



            kernel = WarpspeedKernel(loadFields, storeFields, registers = 128, flops = imageCount*inputChannels*outputChannels*2)

            lc = LaunchConfig.compute(kernel, blockSize, (imageSize[0]-2, imageSize[1]-2, 1), (1,1,1), device)
            basic = BasicMetrics.compute(lc, device, kernel)
            pred = DerivedMetrics(lc, basic, device)

            print(basic)
            print(pred)
