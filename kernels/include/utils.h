#pragma once

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", err_, cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

template <class T>
void calculate_launch_params(int n_threads, int *grid_size, int *block_size, T func) {
    int min_grid_size;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, block_size, func, 0, 0));
    *grid_size = (n_threads + *block_size + 1) / *block_size;
}