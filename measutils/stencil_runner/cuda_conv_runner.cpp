#include "cuda_rtc.hpp"
#include <Python.h>

using namespace std;

extern "C" double timeKernel(const char *kernelText, const char *funcName,
                             int blockSizeX, int blockSizeY, int blockSizeZ,
                             int blockCountX, int blockCountY, int blockCountZ,
                             int input_channels, int output_channels,
                             int batch_size, int width, int height) {

  size_t errorTensorCount =
      (width + 1) * (height + 1) * batch_size * output_channels;
  size_t nextErrorTensorCount =
      (width + 1) * (height + 1) * batch_size * input_channels;
  size_t weightsTensorCount = input_channels * output_channels * 3 * 3;

  init((errorTensorCount + nextErrorTensorCount + weightsTensorCount) *
       sizeof(float));

  float *errorTensor = (float *)src;
  float *nextErrorTensor = ((float *)src) + errorTensorCount;
  float *weightsTensor = nextErrorTensor + nextErrorTensorCount;

  auto kernel = buildKernel(kernelText, funcName);

  void *args[] = {&errorTensor, &nextErrorTensor, &weightsTensor,
                  &batch_size,  &input_channels,  &output_channels,
                  &width,       &height};

  MeasurementSeries times;

  checkCudaErrors(cuLaunchKernel(kernel, blockCountX, blockCountY, blockCountZ,
                                 blockSizeX, blockSizeY, blockSizeZ, 0, NULL,
                                 args, 0));
  checkCudaErrors(cuLaunchKernel(kernel, blockCountX, blockCountY, blockCountZ,
                                 blockSizeX, blockSizeY, blockSizeZ, 0, NULL,
                                 args, 0));
  cudaDeviceSynchronize();
  double totalT1 = dtime();
  while (dtime() - totalT1 < 0.5 || times.count() < 3) {

    cudaDeviceSynchronize();
    double t1 = dtime();
    checkCudaErrors(cuLaunchKernel(kernel, blockCountX, blockCountY,
                                   blockCountZ, blockSizeX, blockSizeY,
                                   blockSizeZ, 0, NULL, args, 0));
    cudaDeviceSynchronize();
    double t2 = dtime();
    times.add(t2 - t1);
  }
  return times.minValue();
}

vector<double> measureMetrics(vector<const char *> metricNames,
                              const char *kernelText, const char *funcName,
                              int blockSizeX, int blockSizeY, int blockSizeZ,
                              int blockCountX, int blockCountY, int blockCountZ,
                              int input_channels, int output_channels,
                              int batch_size, int width, int height) {

  size_t errorTensorCount =
      (width + 1) * (height + 1) * batch_size * output_channels;
  size_t nextErrorTensorCount =
      (width + 1) * (height + 1) * batch_size * input_channels;
  size_t weightsTensorCount = input_channels * output_channels * 3 * 3;

  init((errorTensorCount + nextErrorTensorCount + weightsTensorCount) *
       sizeof(float));

  float *errorTensor = (float *)src;
  float *nextErrorTensor = ((float *)src) + errorTensorCount;
  float *weightsTensor = nextErrorTensor + nextErrorTensorCount;

  auto kernel = buildKernel(kernelText, funcName);

  void *args[] = {&errorTensor, &nextErrorTensor, &weightsTensor,
                  &batch_size,  &input_channels,  &output_channels,
                  &width,       &height};

  cudaDeviceSynchronize();
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
                                      PyObject *blockSize, PyObject *gridSize,
                                      PyObject *args) {

  std::vector<const char *> metricNameVector;

  for (int i = 0; i < PyList_Size(metricNames); i++) {
    char *cstr = (char *)PyUnicode_1BYTE_DATA(PyList_GetItem(metricNames, i));
    metricNameVector.push_back(cstr);
  }

  /*auto values =
      measureMetrics(metricNameVector, (char *)PyUnicode_1BYTE_DATA(kernelText),
                     (char *)PyUnicode_1BYTE_DATA(funcName),
                     PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
                     PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
                     PyLong_AsLong(PyTuple_GetItem(blockSize, 2)),
                     PyLong_AsLong(PyTuple_GetItem(gridSize, 0)),
                     PyLong_AsLong(PyTuple_GetItem(gridSize, 1)),
                     PyLong_AsLong(PyTuple_GetItem(gridSize, 2)),
                     PyLong_AsLong(PyTuple_GetItem(args, 0)),
                     PyLong_AsLong(PyTuple_GetItem(args, 1)),
                     PyLong_AsLong(PyTuple_GetItem(args, 2)),
                     PyLong_AsLong(PyTuple_GetItem(args, 3)),
                     PyLong_AsLong(PyTuple_GetItem(args, 4)));
*/
  auto values =
      measureMetrics(metricNameVector, (char *)PyUnicode_1BYTE_DATA(kernelText),
                     (char *)PyUnicode_1BYTE_DATA(funcName),
                     PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
                     PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
                     PyLong_AsLong(PyTuple_GetItem(blockSize, 2)),
                     PyLong_AsLong(PyTuple_GetItem(gridSize, 0)),
                     PyLong_AsLong(PyTuple_GetItem(gridSize, 1)),
                     PyLong_AsLong(PyTuple_GetItem(gridSize, 2)),
                     PyLong_AsLong(PyTuple_GetItem(args, 0)),
                     PyLong_AsLong(PyTuple_GetItem(args, 1)),
                     PyLong_AsLong(PyTuple_GetItem(args, 2)),
                     PyLong_AsLong(PyTuple_GetItem(args, 3)),
                     PyLong_AsLong(PyTuple_GetItem(args, 4)));

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

    std::cout << "\n";
  }
  return 0;
}
