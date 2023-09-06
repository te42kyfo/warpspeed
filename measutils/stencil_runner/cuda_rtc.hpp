#ifndef CUDA_RTC_H_
#define CUDA_RTC_H_

#include "../gpu_metrics/gpu_metrics.hpp"
#include "MeasurementSeries.hpp"
#include "dtime.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <nvrtc.h>
#include <string>
#include <vector>

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
    }
    checkCudaErrors(cuMemAlloc(&src, newBufferSizeBytes));
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

#endif // CUDA_RTC_H_
