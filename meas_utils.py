#!/usr/bin/env python3
import sys
from subprocess import run, PIPE

from column_print import *
import cProfile
from functools import partial, reduce
from operator import mul
import pycuda.driver as drv

import measure_metric.measureMetric as measureMetric

import math
from timeit import timeit
import pycuda


def timeKernel(run_func):
    start = drv.Event()
    end = drv.Event()
    start.record()
    run_func()
    end.record()
    end.synchronize()
    return end.time_since(start)


def benchKernel(run_func):
    run_func()
    repeats = [timeKernel(run_func) for i in range(0, 5)]
    repeats.sort()
    time = repeats[len(repeats) // 2]
    return time


def measureMetrics(runFunc, size):
    lupCount = reduce(mul, size)

    measurements = []
    for i in range(0, 7):
        measureMetric.measureBandwidthStart()
        runFunc()
        measurements.append(measureMetric.measureMetricStop())

    medians = [
        sorted([m[i] for m in measurements])[len(measurements) // 2] for i in range(len(measurements[0]))
    ]


    return  medians[0] / lupCount, medians[1] / lupCount, medians[2] * 32 / lupCount,  medians[3] * 32 / lupCount, medians[4]*32 / lupCount, medians[5]*32 / lupCount, medians[6]*32 / lupCount, medians[7]*32 / lupCount, medians[8]*32 / lupCount, medians[9] / lupCount






def printSASS(code):
    print(code)
    cubin = pycuda.compiler.compile(code, options=["-w", "-std=c++11"], arch="sm_70")

    run(['echo "' + code + '" >> temp.cubin'], stdout=PIPE, shell=True)

    newFile = open("temp.cubin", "wb")
    newFile.write(cubin)
    newFile.close()

    result = run(["nvdisasm  temp.cubin"], stdout=PIPE, shell=True)

    print(len(result.stdout.decode("utf-8").split("\n")))

    print(result.stdout.decode("utf-8"))

    newFile = open("temp.disasm", "wb")
    newFile.write(result.stdout)
    newFile.close()
