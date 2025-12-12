#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define ROWS 1024
#define COLS 1024

// Shared-memory optimized kernel
__global__ void next_gen_kernel_shared(const int* d_grid, int* d_new_grid) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    const int TILE_W = blockDim.x + 2;  // +2 halo
    extern __shared__ int tile[];

    int local_r = ty + 1;
    int local_c = tx + 1;

    auto idx_global = [] __device__ (int r, int c) {
        r = (r + ROWS) % ROWS;
        c = (c + COLS) % COLS;
        return r * COLS + c;
    };

    if (row < ROWS && col < COLS) {
        tile[local_r * TILE_W + local_c] =
            d_grid[idx_global(row, col)];
    }

    if (tx == 0 && row < ROWS)
        tile[local_r * TILE_W + 0] =
            d_grid[idx_global(row, col - 1)];

    if (tx == blockDim.x - 1 && row < ROWS)
        tile[local_r * TILE_W + (TILE_W - 1)] =
            d_grid[idx_global(row, col + 1)];

    if (ty == 0 && col < COLS)
        tile[0 * TILE_W + local_c] =
            d_grid[idx_global(row - 1, col)];

    if (ty == blockDim.y - 1 && col < COLS)
        tile[(TILE_W - 1) * TILE_W + local_c] =
            d_grid[idx_global(row + 1, col)];

    if (tx == 0 && ty == 0)
        tile[0] = d_grid[idx_global(row - 1, col - 1)];

    if (tx == blockDim.x - 1 && ty == 0)
        tile[TILE_W - 1] = d_grid[idx_global(row - 1, col + 1)];

    if (tx == 0 && ty == blockDim.y - 1)
        tile[(TILE_W - 1) * TILE_W] = d_grid[idx_global(row + 1, col - 1)];

    if (tx == blockDim.x - 1 && ty == blockDim.y - 1)
        tile[(TILE_W - 1) * TILE_W + (TILE_W - 1)] =
            d_grid[idx_global(row + 1, col + 1)];

    __syncthreads();

    if (row >= ROWS || col >= COLS) return;

    int live = 0;
    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue;
            live += tile[(local_r + dr) * TILE_W + (local_c + dc)];
        }
    }

    int cell = tile[local_r * TILE_W + local_c];
    int out_idx = row * COLS + col;

    if (cell == 1)
        d_new_grid[out_idx] = (live == 2 || live == 3) ? 1 : 0;
    else
        d_new_grid[out_idx] = (live == 3) ? 1 : 0;
}

void print_grid(const std::vector<int>& grid) {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            std::cout << (grid[r * COLS + c] ? "o" : ".");
        }
        std::cout << "\n";
    }
}

int main() {
    srand(time(0));

    std::vector<int> h_grid(ROWS * COLS);
    for (int i = 0; i < ROWS * COLS; i++)
        h_grid[i] = rand() % 2;

    int *d_grid, *d_new_grid;
    size_t size = ROWS * COLS * sizeof(int);

    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_new_grid, size);

    cudaMemcpy(d_grid, h_grid.data(), size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((COLS + threads.x - 1) / threads.x,
                (ROWS + threads.y - 1) / threads.y);

    size_t shmem_size = (threads.x + 2) * (threads.y + 2) * sizeof(int);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const int generations = 200;
    for (int gen = 0; gen < generations; gen++) {
        next_gen_kernel_shared<<<blocks, threads, shmem_size>>>(d_grid, d_new_grid);
        std::swap(d_grid, d_new_grid);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_grid.data(), d_grid, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);

std::cout << "GPU Time: " << ms / 1000.0 << " seconds\n";

    print_grid(h_grid);

    cudaFree(d_grid);
    cudaFree(d_new_grid);

    return 0;
}
