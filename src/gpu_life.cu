#include "cuda_utils.h"
#include "life_rules.h"
#include <cuda_runtime.h>
#include <cstdint>

// Naive: each thread reads 8 neighbors directly from global memory.
// This is correct but can be memory-traffic heavy.

__device__ __forceinline__ int wrap_idx(int v, int limit, bool periodic) {
  if (periodic) {
    int x = v % limit;
    if (x < 0) x += limit;
    return x;
  } else {
    // clamp
    if (v < 0) return 0;
    if (v >= limit) return limit - 1;
    return v;
  }
}

__global__ void gol_naive_kernel(const uint8_t* __restrict__ cur,
                                 uint8_t* __restrict__ nxt,
                                 int w, int h,
                                 bool periodic) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // thread handles one x
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // thread handles one y

  if (x >= w || y >= h) return;

  int live = 0;
  // Count neighbors in global memory.
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      int nx = wrap_idx(x + dx, w, periodic);
      int ny = wrap_idx(y + dy, h, periodic);
      live += (cur[ny * w + nx] != 0);
    }
  }

  uint8_t alive = cur[y * w + x];
  nxt[y * w + x] = next_state(alive, live);
}

void gpu_step_naive(const uint8_t* d_cur, uint8_t* d_nxt,
                    int w, int h, bool periodic,
                    int block, cudaStream_t stream) {
  dim3 blockDim(block, block, 1);
  dim3 gridDim((w + blockDim.x - 1) / blockDim.x,
               (h + blockDim.y - 1) / blockDim.y,
               1);

  gol_naive_kernel<<<gridDim, blockDim, 0, stream>>>(d_cur, d_nxt, w, h, periodic);
  CUDA_CHECK(cudaGetLastError());
}