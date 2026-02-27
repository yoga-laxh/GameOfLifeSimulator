#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro to check CUDA calls and fail fast with line info.
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(err__), __FILE__, __LINE__);           \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// GPU timer using cudaEvent for accurate kernel timing.
struct GpuTimer {
  cudaEvent_t start_ev{};
  cudaEvent_t stop_ev{};

  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));
  }

  ~GpuTimer() {
    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
  }

  inline void start(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(start_ev, stream));
  }

  inline float stop(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(stop_ev, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
    return ms;
  }
};