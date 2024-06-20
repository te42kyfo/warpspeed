#!/usr/bin/env python3


class Device:
    def peakFP32(self):
        return self.fp32CycleSM * self.clock * self.smCount * 2

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

    name = "V100"


class DeviceAmpere(DeviceVolta):
    clock = 1.410
    smCount = 108
    sizeL2 = 20 * 1024 * 1024
    sizeL1 = 192 * 1024

    L2BW = 4500
    memBW = 1400

    name = "A100"


class DeviceAmpereA100_80GB(DeviceAmpere):
    memBW = 1600
    L2BW = 4500
    name = "A100_80GB"

    displayName = "A100"


class DeviceAmpereA40(DeviceAmpere):
    clock = 1.74
    smCount = 84

    sizeL2 = 6 * 1024 * 1024
    sizeL1 = 128 * 1024

    subWarpSize = 32
    lsuCount = 16
    fp32CycleSM = 128

    memBW = 670
    L2BW = 2200
    name = "A40"


class DeviceL40(DeviceAmpereA40):
    clock = 2.49
    smCount = 142

    sizeL2 = 96 * 1024 * 1024
    sizeL1 = 128 * 1024

    memBW = 800
    L2BW = 6000

    name = "L40"


class DeviceHopperH200(DeviceAmpere):
    clock = 1.98
    smCount = 132
    memBW = 3800
    L2BW = 10000
    name = "H200"
    displayName = "GH200"

    fp64CycleSM = 64
    fp32CycleSM = 128


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
    sizeL2 = 6 * 1024 * 1024

    L2BW = 2500
    memBW = 1200

    name = "MI100"


class DeviceMI210(DeviceCDNA):
    clock = 1.7
    smCount = 104
    sizeL2 = 6 * 1024 * 1024

    fp64CycleSM = 64

    L2BW = 5000
    memBW = 1400

    name = "MI210"


class DeviceRDNA(Device):
    CLAllocationSize = 128
    CLFetchSize = 64

    warpSize = 32

    fp64CycleSM = 4
    fp32CycleSM = 64

    subWarpSize = 32

    API = "HIP"
    L1Model = "CDNA"


class DeviceRX6900XT(DeviceRDNA):
    sizeL1 = 32 * 1024

    clock = 2.25
    smCount = 80
    sizeL2 = 4 * 1024 * 1024
    L2BW = 2500
    memBW = 550

    name = "RX6900XT"
