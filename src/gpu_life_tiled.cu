#include "cuda_utils.h"
#include "life_rules.h"
#include <cuda_runtime.h>
#include <cstdint>

// Optimized: shared-memory tiling.
// Each block loads a (blockDim.x + 2) x (blockDim.y + 2) tile including halo,
// so neighbor reads mostly come from fast shared memory.

__device__ __forceinline__ int wrap_idx(int v, int limit, bool periodic) {
  if (periodic) {
    int x = v % limit;
    if (x < 0) x += limit;
    return x;
  } else {
    if (v < 0) return 0;
    if (v >= limit) return limit - 1;
    return v;
  }
}

template<int BLOCK>
__global__ void gol_tiled_kernel(const uint8_t* __restrict__ cur,
                                 uint8_t* __restrict__ nxt,
                                 int w, int h,
                                 bool periodic) {
  // Shared tile with halo (BLOCK+2).
  __shared__ uint8_t tile[BLOCK + 2][BLOCK + 2];

  // Global coordinates for this thread's cell.
  int gx = blockIdx.x * BLOCK + threadIdx.x;
  int gy = blockIdx.y * BLOCK + threadIdx.y;

  // Local coordinates in shared tile (offset by +1 for halo border).
  int lx = threadIdx.x + 1;
  int ly = threadIdx.y + 1;

  // Load center cell for threads in-bounds.
  if (gx < w && gy < h) {
    tile[ly][lx] = cur[gy * w + gx];
  } else {
    tile[ly][lx] = 0u; // out-of-range threads set to 0 to avoid garbage
  }

  // Load halo edges: only some threads participate to avoid duplicates.
  // Left halo
  if (threadIdx.x == 0) {
    int nx = wrap_idx(gx - 1, w, periodic);
    int ny = (gy < h) ? gy : wrap_idx(gy, h, periodic);
    tile[ly][0] = (gx < w && gy < h) ? cur[ny * w + nx] : 0u;
  }
  // Right halo
  if (threadIdx.x == BLOCK - 1) {
    int nx = wrap_idx(gx + 1, w, periodic);
    int ny = (gy < h) ? gy : wrap_idx(gy, h, periodic);
    tile[ly][BLOCK + 1] = (gx < w && gy < h) ? cur[ny * w + nx] : 0u;
  }
  // Top halo
  if (threadIdx.y == 0) {
    int nx = (gx < w) ? gx : wrap_idx(gx, w, periodic);
    int ny = wrap_idx(gy - 1, h, periodic);
    tile[0][lx] = (gx < w && gy < h) ? cur[ny * w + nx] : 0u;
  }
  // Bottom halo
  if (threadIdx.y == BLOCK - 1) {
    int nx = (gx < w) ? gx : wrap_idx(gx, w, periodic);
    int ny = wrap_idx(gy + 1, h, periodic);
    tile[BLOCK + 1][lx] = (gx < w && gy < h) ? cur[ny * w + nx] : 0u;
  }

  // Corners (4 threads handle 4 corners)
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int nx = wrap_idx(gx - 1, w, periodic);
    int ny = wrap_idx(gy - 1, h, periodic);
    tile[0][0] = (gx < w && gy < h) ? cur[ny * w + nx] : 0u;
  }
  if (threadIdx.x == BLOCK - 1 && threadIdx.y == 0) {
    int nx = wrap_idx(gx + 1, w, periodic);
    int ny = wrap_idx(gy - 1, h, periodic);
    tile[0][BLOCK + 1] = (gx < w && gy < h) ? cur[ny * w + nx] : 0u;
  }
  if (threadIdx.x == 0 && threadIdx.y == BLOCK - 1) {
    int nx = wrap_idx(gx - 1, w, periodic);
    int ny = wrap_idx(gy + 1, h, periodic);
    tile[BLOCK + 1][0] = (gx < w && gy < h) ? cur[ny * w + nx] : 0u;
  }
  if (threadIdx.x == BLOCK - 1 && threadIdx.y == BLOCK - 1) {
    int nx = wrap_idx(gx + 1, w, periodic);
    int ny = wrap_idx(gy + 1, h, periodic);
    tile[BLOCK + 1][BLOCK + 1] = (gx < w && gy < h) ? cur[ny * w + nx] : 0u;
  }

  __syncthreads(); // Ensure tile is fully loaded before computing.

  if (gx >= w || gy >= h) return;

  // Neighbor sum from shared memory (fast).
  int live =
      (tile[ly - 1][lx - 1] != 0) + (tile[ly - 1][lx] != 0) + (tile[ly - 1][lx + 1] != 0) +
      (tile[ly][lx - 1] != 0)     +                            (tile[ly][lx + 1] != 0) +
      (tile[ly + 1][lx - 1] != 0) + (tile[ly + 1][lx] != 0) + (tile[ly + 1][lx + 1] != 0);

  uint8_t alive = tile[ly][lx];
  nxt[gy * w + gx] = next_state(alive, live);
}

void gpu_step_tiled(const uint8_t* d_cur, uint8_t* d_nxt,
                    int w, int h, bool periodic,
                    int block, cudaStream_t stream) {
  // We implement templated fixed-BLOCK kernels for common sizes for best performance.
  // If you want other sizes, add more cases or write a dynamic shared memory version.
  dim3 blockDim(block, block, 1);
  dim3 gridDim((w + block - 1) / block,
               (h + block - 1) / block,
               1);

  if (block == 8) {
    gol_tiled_kernel<8><<<gridDim, blockDim, 0, stream>>>(d_cur, d_nxt, w, h, periodic);
  } else if (block == 16) {
    gol_tiled_kernel<16><<<gridDim, blockDim, 0, stream>>>(d_cur, d_nxt, w, h, periodic);
  } else if (block == 32) {
    gol_tiled_kernel<32><<<gridDim, blockDim, 0, stream>>>(d_cur, d_nxt, w, h, periodic);
  } else {
    // Fallback: recommend 16 if user passes something else.
    gol_tiled_kernel<16><<<gridDim, dim3(16,16,1), 0, stream>>>(d_cur, d_nxt, w, h, periodic);
  }

  CUDA_CHECK(cudaGetLastError());
}