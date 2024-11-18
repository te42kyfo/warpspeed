#!/usr/bin/env python3


class Device:
    def peakFP32(self):
        return self.fp32CycleSM * self.clock * self.smCount * 2

    def peakFP64(self):
        return self.fp64CycleSM * self.clock * self.smCount * 2

    def getDisplayName(self):

        if hasattr(self, "displayName"):
            return self.displayName
        return self.name


class DeviceVolta(Device):
    CLAllocationSize = 128
    CLFetchSize = 32

    warpSize = 32

    fp64CycleSM = 32
    fp32CycleSM = 64

    API = "CUDA"
    L1Model = "NV"

    subWarpSize = 32
    lsuCount = 32


class DeviceV100(DeviceVolta):
    clock = 1.38
    smCount = 80
    sizeL2 = 6 * 1024 * 1024
    sizeL1 = 128 * 1024

    L2BW = 2500
    memBW = 790
    name = "Tesla V100-PCIE-32GB"
    displayName = "V100"


class DeviceAmpere(DeviceVolta):
    clock = 1.410
    smCount = 108
    sizeL2 = 20 * 1024 * 1024
    sizeL1 = 192 * 1024
    displayName = "Ampere"


class DeviceAmpereA100_40GB(DeviceAmpere):
    L2BW = 4500
    memBW = 1400
    name = "NVIDIA A100-SXM4-40GB"
    displayName = "A100"


class DeviceAmpereA100_80GB(DeviceAmpere):
    memBW = 1600
    L2BW = 4500
    name = "NVIDIA A100-SXM4-80GB"
    displayName = "A100"


class DeviceAmpereA40(DeviceAmpere):
    clock = 1.74
    smCount = 84

    sizeL2 = 6 * 1024 * 1024
    sizeL1 = 128 * 1024

    subWarpSize = 32
    lsuCount = 16

    fp32CycleSM = 64
    fp64CycleSM = 4

    memBW = 670
    L2BW = 2200
    name = "NVIDIA A40"
    displayName = "A40"


class DeviceL40(DeviceAmpereA40):
    clock = 2.49
    smCount = 142

    sizeL2 = 96 * 1024 * 1024
    sizeL1 = 128 * 1024

    memBW = 800
    L2BW = 6100

    fp32CycleSM = 128

    displayName = "L40"
    name = "NVIDIA L40"


class DeviceL40s(DeviceL40):
    clock = 2.52
    smCount = 142

    sizeL2 = 96 * 1024 * 1024
    sizeL1 = 128 * 1024

    memBW = 800
    L2BW = 6120

    displayName = "L40s"
    name = "NVIDIA L40S"


class DeviceHopperH200(DeviceAmpere):
    clock = 1.98
    smCount = 132
    memBW = 3800
    L2BW = 11100
    name = "H200"
    displayName = "GH200"
    name = "NVIDIA GH200 480GB"

    fp64CycleSM = 64
    fp32CycleSM = 128

    sizeL1 = 256 * 1024
    sizeL2 = 30 * 1024 * 1024


class DeviceCDNA(Device):
    CLAllocationSize = 64
    CLFetchSize = 64

    warpSize = 64

    fp64CycleSM = 32
    fp32CycleSM = 64

    sizeL1 = 16 * 1024

    subWarpSize = 16

    API = "HIP"
    L1Model = "CDNA"


class DeviceMI100(DeviceCDNA):
    clock = 1.2
    smCount = 110
    sizeL2 = 8 * 1024 * 1024

    L2BW = 2500
    memBW = 1200

    name = "AMD Instinct MI100"
    displayName = "MI100"


class DeviceMI210(DeviceCDNA):
    clock = 1.7
    smCount = 104
    sizeL2 = 6 * 1024 * 1024

    fp64CycleSM = 64

    L2BW = 4800
    memBW = 1400

    displayName = "MI210"

    name = "AMD Instinct MI210"


class DeviceMI300(DeviceCDNA):
    clock = 2.1
    smCount = 304
    sizeL2 = 4 * 1024 * 1024
    sizeL1 = 32 * 1024

    fp64CycleSM = 128
    fp32CycleSM = 64

    CLAllocationSize = 128
    CLFetchSize = 128

    L2BW = 10800
    memBW = 4100

    displayName = "MI300X"

    name = "AMD Instinct MI300X"


class DeviceRDNA(Device):
    CLAllocationSize = 128
    CLFetchSize = 128

    warpSize = 32

    fp64CycleSM = 4
    fp32CycleSM = 64

    subWarpSize = 32

    API = "HIP"
    L1Model = "CDNA"


class DeviceRX6900XT(DeviceRDNA):
    sizeL1 = 32 * 1024

    clock = 2.56
    smCount = 80
    sizeL2 = 4 * 1024 * 1024
    L2BW = 5000
    memBW = 550

    displayName = "RX6900XT"
    name = "AMD Radeon RX 6900 XT"


deviceList = [
    DeviceAmpereA40(),
    DeviceL40(),
    DeviceL40s(),
    DeviceV100(),
    DeviceAmpereA100_40GB(),
    DeviceAmpereA100_80GB(),
    DeviceHopperH200(),
    DeviceMI100(),
    DeviceMI210(),
    DeviceMI300(),
    DeviceRX6900XT(),
]


def selectDevice(name):
    for d in deviceList:
        if d.name == name:
            return d

    print("No device with name ", name, " in device list!")
    return None


import inspect


tableList = [
    DeviceAmpereA40(),
    DeviceL40s(),
    DeviceV100(),
    DeviceAmpereA100_80GB(),
    DeviceHopperH200(),
    DeviceMI100(),
    DeviceMI210(),
    DeviceMI300(),
    DeviceRX6900XT(),
]


propertyList = []


def getDevicePropertyList():
    for device in deviceList:
        for k, v in inspect.getmembers(device):
            if not str(k).startswith("__") and k not in propertyList:
                propertyList.append(k)

    for prop in propertyList:
        print(prop, end=" & ")
        for device in deviceList:
            if inspect.ismethod(getattr(device, prop, "-")):
                print(getattr(device, prop)(), end=" & ")
            else:
                print(getattr(device, prop, "-"), end=" & ")
        print("\\\\")

    print()

    print("  &", end="")
    for device in tableList:
        print("\\rot{" + device.name + "}", end=" & ")
    print("\\\\")

    propList = [
        ("displayName", "", 0),
        ("API", "", 0),
        ("L1Model", "", 0),
        ("warpSize", "", 0),
        ("subWarpSize", "", 0),
        ("smCount", "", 0),
        ("clock", "GHz", 0),
        ("CLAllocationSize", "\\bytes", 0),
        ("CLFetchSize", "\\bytes", 0),
        ("lsuCount", "", 0),
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
        print("\\rotg{", propName, "} ", end=" & ")

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
        print("\\\\")
