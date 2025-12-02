#include<iostream>
#include<vector>
#include<cstdlib>
#include<ctime>
#include<cuda_runtime.h>

#define rows 1024
#define cols 1024

// Device (GPU) function which is to count the neighnors

__device__ int count_live_neighbors(int* grid, int r, int c){
    int count =0;

    for (int dr= -1; dr <=1; dr++){
        for (int dc= -1; dc <= 1; dc++){
            if (dr ==0 && dc==0) continuel

            int nr= (r +dr + rows) % rows;
            int nc= (c +dc + cols) % cols;

            count += grid[nr * cols + nc];
        }
    }
    return count;
}

//CUDA Kernel for next generation

__global__ void next_gen_kernel(int* d_grid, int* d_new_grid){
    int idx= blockIdx .x * blockDim.x + threadIdx.x;
    int total_cells =rows * cols;

    if (idx >= total_cells) return;

    int r=idx/cols;
    int c=idx % cols;

    int alive =count_live_neighbors(d_grid, r, c);
    int cell = d_grid[idx];

    if (cell==1){
        d_new_grid[idx] = (alive ==2 || alive ==3) ? 1:0;
    }
    else{
        d_new_grid[idx]=(alive == 3)?1 :0;
    }

}

//CPU healper

void print_grid(const std::vector<int> & grid){
    for (int r=0; r <rows; r++){
        for (int c=0; c<cols; c++){
            std::count << (grid[r * cols + c] ? "o":'.');
        }
        std::count << "\n";
    }
}

int main (){
    srand(time(0));
    
    //CPU grid
    std::vector<int> h_grid(rows * cols);
    std::vector<int> h_new_grid(rows * cols);

    //random init

    for(int i=0 ; i < rows * cols; i++){
        h_grid[i]= rand()% 2;
    }

    //GPU memory

    int *d_grid, *d_new_grid;

    size_t size =rows * cols * sizeof(int);

    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_new_grid, size);

    //copy (Host ---> Device)

    cudaMemcpy(d_grid , h_grid.data(), size, cudaMemcpyHostToDevice);

    int treads =256;
    int blocks= (rows * cols + treads -1) / threads;

    const int generations =200;

    for (int gen=0; gen < generations; gen++){
        next_gen_kernel<<<blocks, threads>>>( d_grid, d_new_grid);

        std::swap(d_grid, d_new_grid);
        cudaDeviceSynchronize();
    }

    //copy back (Device ----> Host)

    cudaMemcpy(h_grid.data(), h_grid, size, cudaMemcpyDeviceToHost);

    //Print final result

    print_grid(h_grid);

    cudaFree(d_grid);
    cudaFree(d_new_grid);

    return 0;
}