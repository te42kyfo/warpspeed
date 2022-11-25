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


class MeasStats:

    def add(idx, val):
        MeasStats.maxApe[idx] = max(MeasStats.maxApe[idx], val)
        MeasStats.meanApe[idx] += val
        MeasStats.samples[idx] += 1


    maxApe = [0] * 6
    meanApe = [0] * 6
    samples = [0] * 6

def benchKernel(run_func):
    run_func()
    repeats = [timeKernel(run_func) for i in range(0, 7)]
    repeats.sort()
    MeasStats.add(5, abs(repeats[0] - repeats[-1]) / repeats[0])
    #print( "{:5.2f} {:5.2f}".format( MeasStats.maxApe[5]*100, MeasStats.meanApe[5] / MeasStats.samples[5] * 100))
    time = repeats[len(repeats) // 2]
    return time


def measureMetrics(runFunc, size):
    lupCount = reduce(mul, size)

    measurements = []
    for i in range(0, 7):
        measureMetric.measureMetricStart(
            [
                "dram__bytes_read.sum",
                "dram__bytes_write.sum",
                #"lts__t_sectors_op_read.sum",
                #"lts__t_sectors_op_write.sum",
                "lts__t_sectors_srcunit_tex_op_read.sum",
                "lts__t_sectors_srcunit_tex_op_write.sum",
                #"lts__t_sectors_srcunit_tex.sum",
                #"lts__t_sectors_srcunit_ltcfabric.sum",
                #"lts__t_sectors.sum",
                #"lts__t_tag_requests.sum",
                "l1tex__data_pipe_lsu_wavefronts.sum",
            ]
        )
        runFunc()
        measurements.append(measureMetric.measureMetricStop())

    medians = [
        sorted([m[i] for m in measurements])[len(measurements) // 2]
        for i in range(len(measurements[0]))
    ]




    for i in range(len(measurements[0])):
        sortedRange = sorted( [m[i] for m in measurements] )
        ape = abs(sortedRange[0] - sortedRange[-1]) / medians[i]
        MeasStats.add(i, ape)
        #print( "{:12.2f} {:12.2f} {:5.2f} {:5.2f} {:5.2f}".format( sortedRange[0], sortedRange[-1], ape, MeasStats.maxApe[i]*100, MeasStats.meanApe[i] / MeasStats.samples[i]*100))



    return (
        medians[0] / lupCount,
        medians[1] / lupCount,
        medians[2] * 32 / lupCount,
        medians[3] * 32 / lupCount,
        #medians[7] * 32 / lupCount,
        #medians[8] * 32 / lupCount,
        medians[4] / lupCount,
    )


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
