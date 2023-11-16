#!/usr/bin/env python3


class Device:
    def peakFP32(self):
        return self.fp32CycleSM * self.clock * self.smCount * 2


class DeviceVolta(Device):
    CLAllocationSize = 128
    CLFetchSize = 32

    warpSize = 32

    fp64CycleSM = 32
    fp32CycleSM = 64

    API = "CUDA"
    L1Model = "NV"


class DeviceCDNA(Device):
    CLAllocationSize = 64
    CLFetchSize = 64

    warpSize = 64

    fp64CycleSM = 32
    fp32CycleSM = 64

    sizeL1 = 16 * 1024

    API = "HIP"
    L1Model = "CDNA"


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


class Device2080Ti(DeviceVolta):  # unverified
    clock = 2100
    smCount = 68
    sizeL2 = 5.5 * 1024 * 1024
    sizeL1 = 64 * 1024

    L2BW = 2000
    memBW = 590

    name = "2080Ti"


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


class DeviceRX6900XT(DeviceCDNA):
    CLAllocationSize = 64
    L2FetchSize = 64

    fp64CycleSM = 4
    fp32CycleSM = 64

    sizeL1 = 32 * 1024

    API = "HIP"
    L1Model = "CDNA"

    clock = 2.25
    smCount = 80
    sizeL2 = 4 * 1024 * 1024
    L2BW = 2500
    memBW = 550

    name = "RX6900XT"
