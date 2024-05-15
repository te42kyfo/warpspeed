#ifdef __HIPCC__
#include "hip_rtc.hpp"
#else
#include "cuda_rtc.hpp"
#endif
#include <Python.h>
#include <vector>

#include "gpu-stats.h"

using namespace std;

void **getBufferPointers(PyObject *buffers, PyObject *alignmentBytes) {

  size_t totalBufferSizeBytes = 0;
  for (int i = 0; i < PyList_Size(buffers); i++) {
    totalBufferSizeBytes += PyLong_AsLong((PyList_GetItem(buffers, i))) + 1024;
  }

  init(totalBufferSizeBytes);
  void **bufferPointers =
      (void **)malloc(PyList_Size(buffers) * sizeof(void *));
  void **bufferPointersPointers =
      (void **)malloc(PyList_Size(buffers) * sizeof(void *));

  void *currPtr = (char *)src;

  for (int i = 0; i < PyList_Size(buffers); i++) {
    bufferPointers[i] = (char *)currPtr + 1024 - PyLong_AsLong(alignmentBytes);
    currPtr =
        (char *)currPtr + PyLong_AsLong((PyList_GetItem(buffers, i))) + 1024;
    bufferPointersPointers[i] = &bufferPointers[i];
  }
  return bufferPointersPointers;
}

extern "C" PyObject *
pyTimeNBufferKernel(PyObject *kernelText, PyObject *funcName,
                    PyObject *blockSize, PyObject *blockCount,
                    PyObject *buffers, PyObject *alignmentBytes) {

  void **bufferPointersPointers = getBufferPointers(buffers, alignmentBytes);

  auto kernel = buildKernel((char *)PyUnicode_1BYTE_DATA(kernelText),
                            (char *)PyUnicode_1BYTE_DATA(funcName));

  MeasurementSeries times;

  double minRunTime = 0.1;

  checkCuErrors(cuLaunchKernel(kernel,
                               PyLong_AsLong(PyTuple_GetItem(blockCount, 0)),
                               PyLong_AsLong(PyTuple_GetItem(blockCount, 1)),
                               PyLong_AsLong(PyTuple_GetItem(blockCount, 2)),
                               PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
                               PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
                               PyLong_AsLong(PyTuple_GetItem(blockSize, 2)), 0,
                               NULL, (void **)bufferPointersPointers, 0));

  int iterations = 1;
  double dt = 0;

  while (dt < minRunTime) {
    iterations *= 2;
    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();
    for (int i = 0; i < iterations; i++) {
      checkCuErrors(
          cuLaunchKernel(kernel, PyLong_AsLong(PyTuple_GetItem(blockCount, 0)),
                         PyLong_AsLong(PyTuple_GetItem(blockCount, 1)),
                         PyLong_AsLong(PyTuple_GetItem(blockCount, 2)),
                         PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
                         PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
                         PyLong_AsLong(PyTuple_GetItem(blockSize, 2)), 0, NULL,
                         (void **)bufferPointersPointers, 0));
    }
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    dt = t2 - t1;
    std::cout << iterations << " " << dt << "\n";
  }

  for (int n = 0; n < 3; n++) {
    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();
    for (int i = 0; i < iterations; i++) {
      checkCuErrors(
          cuLaunchKernel(kernel, PyLong_AsLong(PyTuple_GetItem(blockCount, 0)),
                         PyLong_AsLong(PyTuple_GetItem(blockCount, 1)),
                         PyLong_AsLong(PyTuple_GetItem(blockCount, 2)),
                         PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
                         PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
                         PyLong_AsLong(PyTuple_GetItem(blockSize, 2)), 0, NULL,
                         (void **)bufferPointersPointers, 0));
    }
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    times.add((t2 - t1) / iterations);
    std::cout << (t2 - t1) / iterations << "\n";
  }

  GPU_ERROR(cudaDeviceSynchronize());
  double start = dtime();
  for (size_t iter = 0; iter < iterations; iter++) {
    checkCuErrors(cuLaunchKernel(kernel,
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 0)),
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 1)),
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 2)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 2)),
                                 0, NULL, (void **)bufferPointersPointers, 0));
  }
  MeasurementSeries powerSeries;
  MeasurementSeries clockSeries;

  int deviceId = 0;

  for (int i = 0; i < 11; i++) {
    usleep(minRunTime * 1e6 / 20);
    auto stats = getGPUStats(deviceId);
    powerSeries.add(stats.power);
    clockSeries.add(stats.clock);
  }

  GPU_ERROR(cudaDeviceSynchronize());

  std::cout << powerSeries.median() << "\n";
  std::cout << clockSeries.median() << "\n";

  free(bufferPointersPointers[0]);
  free(bufferPointersPointers);

  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject *resultTuple = PyTuple_New(3);
  PyTuple_SetItem(resultTuple, 0, PyFloat_FromDouble(times.minValue()));
  PyTuple_SetItem(resultTuple, 1, PyFloat_FromDouble(clockSeries.median()));
  PyTuple_SetItem(resultTuple, 2,
                  PyFloat_FromDouble(powerSeries.maxValue() / 1000));

  PyGILState_Release(gstate);
  return resultTuple;
}

extern "C" PyObject *
pyMeasureMetricsNBufferKernel(PyObject *metricNames, PyObject *kernelText,
                              PyObject *funcName, PyObject *blockSize,
                              PyObject *blockCount, PyObject *buffers,
                              PyObject *alignmentBytes) {

  std::vector<const char *> metricNameVector;

  for (int i = 0; i < PyList_Size(metricNames); i++) {
    char *cstr = (char *)PyUnicode_1BYTE_DATA(PyList_GetItem(metricNames, i));
    metricNameVector.push_back(cstr);
  }

  void **bufferPointersPointers = getBufferPointers(buffers, alignmentBytes);

  auto kernel = buildKernel((char *)PyUnicode_1BYTE_DATA(kernelText),
                            (char *)PyUnicode_1BYTE_DATA(funcName));

  measureMetricsStart(metricNameVector);
  checkCuErrors(cuLaunchKernel(kernel,
                               PyLong_AsLong(PyTuple_GetItem(blockCount, 0)),
                               PyLong_AsLong(PyTuple_GetItem(blockCount, 1)),
                               PyLong_AsLong(PyTuple_GetItem(blockCount, 2)),
                               PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
                               PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
                               PyLong_AsLong(PyTuple_GetItem(blockSize, 2)), 0,
                               NULL, (void **)bufferPointersPointers, 0));
  auto values = measureMetricStop();

  std::cout << values.size();

  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject *result = PyList_New(0);
  for (auto value : values) {
    PyList_Append(result, PyFloat_FromDouble(value));
  }

  PyGILState_Release(gstate);

  return result;
}
