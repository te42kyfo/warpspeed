#include "cuda_rtc.hpp"
#include <Python.h>
#include <vector>

using namespace std;

extern "C" double timeKernel(const char *kernelText, const char *funcName,
                             int blockSizeX, int blockSizeY, int blockSizeZ,
                             int blockCountX, int blockCountY, int blockCountZ,
                             size_t bufferSizeBytes, size_t alignmentBytes) {

  init(bufferSizeBytes);

  auto kernel = buildKernel(kernelText, funcName);

  std::cout << alignmentBytes << " alignment Bytes\n";
  void *alignedSrc = (char *)src + 1024 - alignmentBytes;
  void *alignedDst = (char *)dst + 1024 - alignmentBytes;
  void *args[] = {&alignedSrc, &alignedDst};

  MeasurementSeries times;

  cudaDeviceSynchronize();
  for (int i = 0; i < 11; i++) {
    cudaDeviceSynchronize();
    double t1 = dtime();
    checkCudaErrors(cuLaunchKernel(kernel, blockCountX, blockCountY,
                                   blockCountZ, blockSizeX, blockSizeY,
                                   blockSizeZ, 0, NULL, args, 0));
    cudaDeviceSynchronize();
    double t2 = dtime();
    times.add(t2 - t1);
  }

  return times.median();
}

#ifndef NO_PYTHON
extern "C" double pyTimeNBufferKernel(PyObject *kernelText, PyObject *funcName,
                                      PyObject *blockSize, PyObject *blockCount,
                                      PyObject *buffers,
                                      PyObject *alignmentBytes) {

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

  auto kernel = buildKernel((char *)PyUnicode_1BYTE_DATA(kernelText),
                            (char *)PyUnicode_1BYTE_DATA(funcName));

  MeasurementSeries times;

  cudaDeviceSynchronize();
  for (int i = 0; i < 11; i++) {
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

    /*checkCudaErrors(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0,
                                   (void **)bufferPointersPointers, nullptr));*/
    cudaDeviceSynchronize();
    double t2 = dtime();
    times.add(t2 - t1);
  }

  free(bufferPointers);
  free(bufferPointersPointers);

  return times.median();
  return 1.0f;
}
#endif

vector<double> measureMetrics(vector<const char *> metricNames,
                              const char *kernelText, const char *funcName,
                              int blockSizeX, int blockSizeY, int blockSizeZ,
                              int blockCountX, int blockCountY, int blockCountZ,
                              size_t bufferSizeBytes, size_t alignmentBytes) {
  init(bufferSizeBytes);

  auto kernel = buildKernel(kernelText, funcName);

  std::cout << alignmentBytes << " alignment Bytes\n";
  void *alignedSrc = (char *)src + 1024 - alignmentBytes;
  void *alignedDst = (char *)dst + 1024 - alignmentBytes;
  void *args[] = {&alignedSrc, &alignedDst};
  measureMetricsStart(metricNames);

  checkCudaErrors(cuLaunchKernel(kernel, blockCountX, blockCountY, blockCountZ,
                                 blockSizeX, blockSizeY, blockSizeZ, 0, NULL,
                                 args, 0));

  auto values = measureMetricStop();
  return values;
}

#ifndef NO_PYTHON
extern "C" PyObject *pyMeasureMetrics(PyObject *metricNames,
                                      PyObject *kernelText, PyObject *funcName,
                                      PyObject *blockSize, PyObject *blockCount,
                                      PyObject *bufferSizeBytes,
                                      PyObject *alignmentBytes) {

  std::vector<const char *> metricNameVector;

  for (int i = 0; i < PyList_Size(metricNames); i++) {
    char *cstr = (char *)PyUnicode_1BYTE_DATA(PyList_GetItem(metricNames, i));
    metricNameVector.push_back(cstr);
  }

  auto values = measureMetrics(
      metricNameVector, (char *)PyUnicode_1BYTE_DATA(kernelText),
      (char *)PyUnicode_1BYTE_DATA(funcName),
      PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
      PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
      PyLong_AsLong(PyTuple_GetItem(blockSize, 2)),
      PyLong_AsLong(PyTuple_GetItem(blockCount, 0)),
      PyLong_AsLong(PyTuple_GetItem(blockCount, 1)),
      PyLong_AsLong(PyTuple_GetItem(blockCount, 2)),
      PyLong_AsLong(bufferSizeBytes), PyLong_AsLong(alignmentBytes));

  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject *result = PyList_New(0);
  for (auto value : values) {
    PyList_Append(result, PyFloat_FromDouble(value));
  }

  PyGILState_Release(gstate);

  return result;
}
#endif
int main(int argc, char **) {

  setenv("HSA_TOOLS_LIB", "/opt/rocm/rocprofiler/lib/librocprofiler64.so", 1);
  setenv("ROCP_METRICS", "/opt/rocm/lib/rocprofiler/metrics.xml", 1);
  setenv("ROCP_HSA_INTERCEPT", "1", 1);

  for (size_t i = 1; i < (size_t)8 * 1024 * 1024 * 1024; i *= 2) {
    string codeString = "extern \"C\" __global__ void updateKernel(double* "
                        "A, double* B) {int "
                        "tidx = threadIdx.x + blockDim.x * blockIdx.x;";
    codeString += "size_t elementCount = " + to_string(i) + ";\n";
    codeString += "for(size_t i = tidx; i < elementCount; i += blockDim.x * "
                  "gridDim.x) {A[i] = 0.2 * A[i];}}";

    double dt = timeKernel(codeString.c_str(), "updateKernel", 1024, 1, 1, 10,
                           1, 1, i * sizeof(double), 0);

    auto vals = measureMetrics({"FETCH_SIZE", "WRITE_SIZE"}, codeString.c_str(),
                               "updateKernel", 1024, 1, 1, 10, 1, 1,
                               i * sizeof(double), 0);

    std::cout << std::setw(10) << i << " " << std::setprecision(0) << std::fixed
              << std::setw(7) << dt * 1000000 << " " << std::setw(7)
              << std::setprecision(4) << dt * 1.4e9 / i << " ";
    for (auto v : vals) {
      std::cout << v * 1024 / i << " B/thread ";
    }
    std::cout << "\n";
  }
  return 0;
}
