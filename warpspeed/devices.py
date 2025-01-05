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

    L1BankWidth = 128
    CLAllocationSize = 128
    CLFetchSize = 32
    CLWriteSize = 32
    DRAMFetchSize = 32

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

    L2BW = 2300
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
    L2BW = 4000
    memBW = 1400
    name = "NVIDIA A100-SXM4-40GB"
    displayName = "A100"


class DeviceAmpereA100_80GB(DeviceAmpere):
    memBW = 1600
    L2BW = 4000
    name = "NVIDIA A100-SXM4-80GB"
    displayName = "A100"


class DeviceAmpereA40(DeviceAmpere):
    clock = 1.74
    smCount = 84

    sizeL2 = 6 * 1024 * 1024
    sizeL1 = 128 * 1024

    subWarpSize = 32
    lsuCount = 16

    fp32CycleSM = 128
    fp64CycleSM = 2

    memBW = 670
    L2BW = 2900
    name = "NVIDIA A40"
    displayName = "A40"


class DeviceL40(DeviceAmpereA40):
    clock = 2.49
    smCount = 142

    DRAMFetchSize = 32

    sizeL2 = 96 * 1024 * 1024
    sizeL1 = 128 * 1024

    memBW = 800
    L2BW = 4300

    displayName = "L40"
    name = "NVIDIA L40"


class DeviceL40s(DeviceL40):
    clock = 2.52
    smCount = 142

    sizeL2 = 96 * 1024 * 1024
    sizeL1 = 128 * 1024

    memBW = 780
    L2BW = 4300

    displayName = "L40s"
    name = "NVIDIA L40S"


class DeviceHopperH200(DeviceAmpere):
    clock = 1.98
    smCount = 132
    memBW = 3200
    L2BW = 6900
    name = "H200"
    displayName = "GH200"
    name = "NVIDIA GH200 480GB"

    fp64CycleSM = 64
    fp32CycleSM = 128

    sizeL1 = 256 * 1024
    sizeL2 = 30 * 1024 * 1024


class DeviceCDNA(Device):
    L1BankWidth = 64
    CLAllocationSize = 64
    CLFetchSize = 64
    CLWriteSize = 64

    DRAMFetchSize = 32

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

    L2BW = 2900
    memBW = 1400

    displayName = "MI210"

    name = "AMD Instinct MI210"


class DeviceMI300(DeviceCDNA):
    clock = 2.1
    smCount = 304
    sizeL2 = 4 * 1024 * 1024
    sizeL1 = 32 * 1024

    fp64CycleSM = 64
    fp32CycleSM = 64

    CLAllocationSize = 128
    CLFetchSize = 128
    CLWriteSize = 64

    L2BW = 13700
    memBW = 4000

    displayName = "MI300X"
    name = "AMD Instinct MI300X"


class DeviceMI300_CPX(DeviceMI300):
    smCount = 38

    displayName = "MI300X-CPX"
    name = "AMD Instinct MI300X-CPX"


class DeviceMI300A(DeviceMI300):
    smCount = 228

    L2BW = 10100
    memBW = 3000

    displayName = "MI300A"
    name = "AMD Instinct MI300A"


class DeviceMI300A_CPX(DeviceMI300A):
    smCount = 38

    displayName = "MI300A-CPX"
    name = "AMD Instinct MI300A-CPX"


class DeviceRDNA(Device):
    L1BankWidth = 128
    CLAllocationSize = 128

    CLFetchSize = 128
    CLWriteSize = 64

    DRAMFetchSize = 32

    warpSize = 32

    fp64CycleSM = 4
    fp32CycleSM = 64

    subWarpSize = 32

    API = "HIP"
    L1Model = "RDNA"


class DeviceRX6900XT(DeviceRDNA):
    sizeL1 = 32 * 1024

    clock = 2.56
    smCount = 80
    sizeL2 = 4 * 1024 * 1024
    L2BW = 2800
    memBW = 520

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
    DeviceMI300_CPX(),
    DeviceMI300A(),
    DeviceMI300A_CPX(),
    DeviceRX6900XT(),
]


def selectDevice(name):
    for d in deviceList:
        if d.name == name:
            return d

    print("No device with name ", name, " in device list!")
    return None
