#ifdef __NVCC__
#include "cuda_rtc.hpp"
#else
#include "hip_rtc.hpp"
#endif
#include <Python.h>
#include <vector>

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

extern "C" double pyTimeNBufferKernel(PyObject *kernelText, PyObject *funcName,
                                      PyObject *blockSize, PyObject *blockCount,
                                      PyObject *buffers,
                                      PyObject *alignmentBytes) {

  void **bufferPointersPointers = getBufferPointers(buffers, alignmentBytes);

  auto kernel = buildKernel((char *)PyUnicode_1BYTE_DATA(kernelText),
                            (char *)PyUnicode_1BYTE_DATA(funcName));

  MeasurementSeries times;

  cudaDeviceSynchronize();
  double totalT1 = dtime();
  while (dtime() - totalT1 < 0.5 || times.count() < 3) {

    cudaDeviceSynchronize();
    double t1 = dtime();
    checkCudaErrors(
        cuLaunchKernel(kernel, PyLong_AsLong(PyTuple_GetItem(blockCount, 0)),
                       PyLong_AsLong(PyTuple_GetItem(blockCount, 1)),
                       PyLong_AsLong(PyTuple_GetItem(blockCount, 2)),
                       PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
                       PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
                       PyLong_AsLong(PyTuple_GetItem(blockSize, 2)), 0, NULL,
                       (void **)bufferPointersPointers, 0));

    cudaDeviceSynchronize();
    double t2 = dtime();
    times.add(t2 - t1);
  }

  free(bufferPointersPointers[0]);
  free(bufferPointersPointers);

  return times.minValue();
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
  checkCudaErrors(cuLaunchKernel(kernel,
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 0)),
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 1)),
                                 PyLong_AsLong(PyTuple_GetItem(blockCount, 2)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
                                 PyLong_AsLong(PyTuple_GetItem(blockSize, 2)),
                                 0, NULL, (void **)bufferPointersPointers, 0));
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
