#include "MeasurementSeries.hpp"
#include "dtime.hpp"
#include <Python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <nvrtc.h>
#include <string>
#include <vector>
#include "../gpu_metrics/gpu_metrics.hpp"
using namespace std;

#ifdef __HIP__
#define checkCudaErrors(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != hipSuccess) {
    std::cerr << "GPUassert: \"" << hipGetErrorString(code) << "\"  in " << file
              << ": " << line << "\n";
    if (abort)
      exit(code);
  }
}
#else

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *errorStr = NULL;
    cuGetErrorString(err, &errorStr);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
  }
}

#endif

#define NVRTC_SAFE_CALL(x)                                                     \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      std::cerr << "\nerror: " #x " failed with error "                        \
                << nvrtcGetErrorString(result) << '\n';                        \
    }                                                                          \
  } while (0)

CUdeviceptr src = 0;
CUdeviceptr dst = 0;
size_t bufferSizeBytes = 0;
CUdevice cuDevice;
CUcontext context;
bool initialized = false;

void init(size_t newBufferSizeBytes) {
  if (!initialized) {
    //checkCudaErrors(cuInit(0));
    //checkCudaErrors(cuDeviceGet(&cuDevice, 0));
    //checkCudaErrors(cuCtxCreate(&context, 0, cuDevice));
    initialized = true;
  }
  if (bufferSizeBytes < newBufferSizeBytes) {

    newBufferSizeBytes = std::max(
        (size_t)1024 * 1024,
        std::max(newBufferSizeBytes, std::min((size_t)1024 * 1024 * 1024 * 10,
                                              (size_t)newBufferSizeBytes * 2)));

    std::cout << "allocate " << newBufferSizeBytes << ", was "
              << bufferSizeBytes << "\n";
    if (src != 0) {
      checkCudaErrors(cuMemFree(src));
      checkCudaErrors(cuMemFree(dst));
    }
    checkCudaErrors(cuMemAlloc(&src, newBufferSizeBytes));
    checkCudaErrors(cuMemAlloc(&dst, newBufferSizeBytes));
    bufferSizeBytes = newBufferSizeBytes;
  }
}

std::map<std::string, CUfunction> funcCache;

CUfunction buildKernel(const char *kernelText, const char *funcName) {

  if (auto it = funcCache.find(std::string(kernelText));
      it == funcCache.end()) {
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernelText, NULL, 0, NULL, NULL));

    const char *opts[] = {"-arch=sm_80"};
    NVRTC_SAFE_CALL(nvrtcCompileProgram(prog, 1, opts));

    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));

    if (logSize > 1) {
      std::cout << log << "\n";
      std::cout << kernelText << "\n";
      std::cout << funcName << "\n";
    }
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

    // std::cout << ptx << "\n";

    CUmodule module;
    CUfunction kernel;
    checkCudaErrors(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    checkCudaErrors(cuModuleGetFunction(&kernel, module, funcName));

    funcCache.insert({std::string(kernelText), kernel});
    // NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
  }

  return funcCache[kernelText];
}

extern "C" double timeKernel(const char *kernelText, const char *funcName,
                             int blockSizeX, int blockSizeY, int blockSizeZ,
                             int blockCountX, int blockCountY, int blockCountZ,
                             size_t bufferSize) {

  init(bufferSize);

  auto kernel = buildKernel(kernelText, funcName);

  void *args[] = {&src, &dst};

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


vector<double> measureMetrics(vector<string> metricNames,
                              const char *kernelText, const char *funcName,
                              int blockSizeX, int blockSizeY, int blockSizeZ,
                              int blockCountX, int blockCountY, int blockCountZ,
                              size_t bufferSize) {
  init(bufferSize);


  auto kernel = buildKernel(kernelText, funcName);
  void *args[] = {&src, &dst};
  measureBandwidthStart();

  checkCudaErrors(cuLaunchKernel(kernel, blockCountX, blockCountY,
                                 blockCountZ, blockSizeX, blockSizeY,
                                 blockSizeZ, 0, NULL, args, 0));


  auto values =  measureMetricStop();
  std::cout << "measureMetrics : ";
  for (auto v : values)
    std::cout << v << " ";
  std::cout << "\n";
  return values;
}

#ifndef NO_PYTHON
extern "C" PyObject *pyMeasureMetrics(PyObject *metricNames,
                                      PyObject *kernelText, PyObject *funcName,
                                      PyObject *blockSize, PyObject *blockCount,
                                      PyObject *bufferSize) {

  std::vector<std::string> metricNameVector;

  for (int i = 0; i < PyList_Size(metricNames); i++) {
    char *cstr = (char *)PyUnicode_1BYTE_DATA(PyList_GetItem(metricNames, i));
    metricNameVector.push_back(std::string(cstr));
  }

  auto values = measureMetrics(
      metricNameVector, (char *)PyUnicode_1BYTE_DATA(kernelText),
      (char *)PyUnicode_1BYTE_DATA(funcName),
      PyLong_AsLong(PyTuple_GetItem(blockSize, 0)),
      PyLong_AsLong(PyTuple_GetItem(blockSize, 1)),
      PyLong_AsLong(PyTuple_GetItem(blockSize, 2)),
      PyLong_AsLong(PyTuple_GetItem(blockCount, 0)),
      PyLong_AsLong(PyTuple_GetItem(blockCount, 1)),
      PyLong_AsLong(PyTuple_GetItem(blockCount, 2)), PyLong_AsLong(bufferSize));

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


  for (int i = 1; i < 1024 * 1024 * 1024; i *= 2) {
    string codeString = "extern \"C\" __global__ void updateKernel(double* A, double* B) {int tidx = threadIdx.x + blockDim.x * blockIdx.x;";
    codeString += "int elementCount = " + to_string(i) + ";\n";
    codeString +=  "for(size_t i = tidx; i < elementCount; i += blockDim.x * gridDim.x) {A[i] = 0.2 * A[i];}}";

    double dt = timeKernel(codeString.c_str(), "updateKernel", 1024, 1, 1, 10, 1, 1, i*sizeof(double));

    auto vals = measureMetrics({"FETCH_SIZE", "WRITE_SIZE"},
        codeString.c_str(), "updateKernel", 1024, 1, 1, i, 10, 1, i*sizeof(double));

    std::cout << std::setw(10) << i << " " << std::setprecision(0) << std::fixed
              << std::setw(7) << dt * 1000000 << " " << std::setw(7)
              << std::setprecision(4) << dt * 1.4e9 / i << " ";
    for(auto v : vals) {
      std::cout << v << " ";
    }
    std::cout << "\n";

  }
  return 0;
}
