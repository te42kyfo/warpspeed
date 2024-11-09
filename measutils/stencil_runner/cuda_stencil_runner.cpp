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

  // std::cout << (char *)PyUnicode_1BYTE_DATA(kernelText) << "\n";
  // std::cout << (char *)PyUnicode_1BYTE_DATA(funcName) << "\n";

  auto kernel = buildKernel((char *)PyUnicode_1BYTE_DATA(kernelText),
                            (char *)PyUnicode_1BYTE_DATA(funcName));

  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  MeasurementSeries times;
  float totalMilliseconds = 0;

  while (totalMilliseconds < 1000) {
    GPU_ERROR(cudaEventRecord(start));
    checkCuErrors(cuLaunchKernel(kernel,
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 0)),
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 1)),
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 2)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 2)),
                                 0, NULL, (void **)bufferPointersPointers, 0));

    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    GPU_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    times.add(milliseconds / 1000);
    totalMilliseconds += milliseconds;
  }

  // power/clock measurement loop running at least twice / or for 2 seconds
  GPU_ERROR(cudaEventRecord(start));
  for (int iter = 0; iter < (int)max(2.0, 2.0 / times.minValue()); iter++) {
    checkCuErrors(cuLaunchKernel(kernel,
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 0)),
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 1)),
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 2)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 2)),
                                 0, NULL, (void **)bufferPointersPointers, 0));
  }
  GPU_ERROR(cudaEventRecord(stop));

  MeasurementSeries powerSeries;
  MeasurementSeries clockSeries;

  int deviceId = 0;
  GPU_ERROR(cudaGetDevice(&deviceId));
  do {
    usleep(1000);
    auto stats = getGPUStats(deviceId);
    powerSeries.add(stats.power);
    clockSeries.add(stats.clock);
  } while (cudaEventQuery(stop) == cudaErrorNotReady);

  std::cout << times.count() << " " << times.median() * 1000 << " ms\n";
  std::cout << powerSeries.count() << " " << powerSeries.median() / 1000
            << " W\n";
  std::cout << clockSeries.count() << " " << clockSeries.median() << " Mhz\n";

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
  auto values = measureMetricsStop();

  // std::cout << values.size();

  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject *result = PyList_New(0);
  for (auto value : values) {
    PyList_Append(result, PyFloat_FromDouble(value));
  }

  PyGILState_Release(gstate);

  return result;
}

extern "C" PyObject *pyGetDeviceName(PyObject *deviceNumber) {

  PyGILState_STATE gstate = PyGILState_Ensure();
  int deviceId = PyLong_AsLong(deviceNumber);
  // GPU_ERROR(cudaSetDevice(deviceId));

  cudaDeviceProp prop;
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;

  PyObject *result = PyUnicode_FromString(deviceName.c_str());

  PyGILState_Release(gstate);

  return result;
}

extern "C" void pyMeasureMetricInit() { initMeasureMetric(); }
