#!/usr/bin/env python3


import inspect
from devices import *

tableList = [
    DeviceAmpereA40(),
    DeviceL40s(),
    DeviceV100(),
    DeviceAmpereA100_80GB(),
    DeviceHopperH200(),
    DeviceMI100(),
    DeviceMI210(),
    DeviceMI300(),
    # DeviceMI300A(),
    DeviceRX6900XT(),
]


def getDevicePropertyList():

    print("  &", end="")
    for device in tableList:
        print("\\rot[40]{" + device.displayName + "}", end=" & ")
    print("\\\\")

    propList = [
        # ("displayName", "", 0),
        ("API", "", 0),
        ("L1Model", "", 0),
        ("warpSize", "", 0),
        ("subWarpSize", "", 0),
        ("smCount", "", 0),
        ("clock", "GHz", 0),
        ("CLAllocationSize", "\\bytes", 0),
        ("CLFetchSize", "\\bytes", 0),
        ("CLWriteSize", "\\bytes", 0),
        ("DRAMFetchSize", "\\bytes", 0),
        ("lsuCount", "", 0),
        ("L1BankWidth", "\\bytes", 0),
        ("sizeL1", "\\KB", 1024),
        ("sizeL2", "\\MB", 1024 * 1024),
        ("fp32CycleSM", "", 0),
        ("fp64CycleSM", "", 0),
        ("peakFP32", "\\TFS", 1000),
        ("peakFP64", "\\TFS", 1000),
        ("L2BW", "\\TBS", 1000),
        ("memBW", "\\TBS", 1000),
    ]
    for prop in propList:
        propName = prop[0]
        propUnit = prop[1]
        print("\\rotg[0]{", propName, "} ", end=" & ")

        prevPropString = ""
        prevCounter = 0

        for device in tableList:

            if inspect.ismethod(getattr(device, propName, "-")):
                val = getattr(device, propName)()
            else:
                val = getattr(device, propName, "-")
            if prop[2] != 0:
                val /= prop[2]
                if val - int(val) > 0.1:
                    if val > 1:
                        propString = "{:.1f}".format(val)
                    else:
                        propString = "{:.2f}".format(val)
                else:
                    propString = "{:.0f}".format(val)
            else:
                propString = val

            if prevPropString != propString and prevPropString != "":
                print(
                    "\\multicolumn{",
                    prevCounter,
                    "}{W}{",
                    prevPropString,
                    "}",
                    end=" & ",
                )
                prevCounter = 1
                prevPropString = propString
            else:
                prevPropString = propString
                prevCounter += 1

        print(
            "\\multicolumn{",
            prevCounter,
            "}{W}{",
            prevPropString,
            "}",
            end=" ",
        )
        print(" & \\cellcolor{white}", propUnit, end="")
        if prop != propList[-1]:
            print("\\\\")


getDevicePropertyList()
