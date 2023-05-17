#include "../gpu_metrics/gpu_metrics.hpp"
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
CUcontext context;
bool initialized = false;

void init(size_t newBufferSizeBytes) {
  bufferSizeBytes += 1024;
  setenv("HSA_TOOLS_LIB", "/opt/rocm/rocprofiler/lib/librocprofiler64.so", 1);
  setenv("ROCP_METRICS", "/opt/rocm/lib/rocprofiler/metrics.xml", 1);
  setenv("ROCP_HSA_INTERCEPT", "1", 1);
  if (!initialized) {
    checkCudaErrors(cuInit(0));
    CUdevice cuDevice;
    checkCudaErrors(cuDeviceGet(&cuDevice, 0));
    checkCudaErrors(cuCtxCreate(&context, 0, cuDevice));
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

#ifdef __NVCC__
    const char *opts[] = {"-arch=sm_80"};
#else
    const char *opts[] = {};
#endif
    NVRTC_SAFE_CALL(
        nvrtcCompileProgram(prog, sizeof(opts) / sizeof(char *), opts));

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
                             size_t bufferSizeBytes, size_t alignmentBytes) {

  init(bufferSizeBytes);

  auto kernel = buildKernel(kernelText, funcName);

  std::cout << alignmentBytes << " alignment Bytes\n";
  void* alignedSrc = (char *)src + 1024 - alignmentBytes;
  void* alignedDst = (char *)dst + 1024 - alignmentBytes;
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

vector<double> measureMetrics(vector<const char *> metricNames,
                              const char *kernelText, const char *funcName,
                              int blockSizeX, int blockSizeY, int blockSizeZ,
                              int blockCountX, int blockCountY, int blockCountZ,
                              size_t bufferSizeBytes, size_t alignmentBytes) {
  init(bufferSizeBytes);


  auto kernel = buildKernel(kernelText, funcName);

  std::cout << alignmentBytes << " alignment Bytes\n";
  void* alignedSrc = (char *)src + 1024 - alignmentBytes;
  void* alignedDst = (char *)dst + 1024 - alignmentBytes;
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
    string codeString =
        "extern \"C\" __global__ void updateKernel(double* A, double* B) {int "
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
