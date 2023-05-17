#ifndef GPU_MEASURE_METRICS_H_
#define GPU_MEASURE_METRICS_H_



#ifdef __HIP__
#include "rocm_metrics/rocm_metrics.hpp"
#else
#include "cuda_metrics/measureMetricPW.hpp"
#endif

#endif // GPU_MEASURE_METRICS_H_
