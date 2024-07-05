#include <iostream>
#include <vector>
#include <chrono>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "sh.cuh"
#include "histogram.cuh"
#include "transforms.cuh"
#include "helper_math.h"
#include "common.h"
#include "math_constants.h"
#include "utils.h"

#ifndef NDEBUG
#include <cuda_profiler_api.h>
#endif

namespace nb = nanobind;

__global__
void dot_product_half_lookup(
    float3 *normals, float3 *wi_world,
    float *bsdf_coeffs, float3 *emitter_coeffs,
    float3 *output,
    int n_rows, int n_cols, int sh_order
) {
    // Get the flattened pixel index and 2d position
    uint2 block_start = make_uint2(blockDim.x * blockIdx.x, blockDim.y * blockIdx.y);
    uint2 block_pos = make_uint2(threadIdx.x, threadIdx.y);
    uint2 pixel_pos = block_start + block_pos;
    uint pixel_idx = pixel_pos.x * n_cols + pixel_pos.y;

    // Make sure we are not out of bounds
    if (pixel_idx >= n_rows * n_cols) return;

    int max_idx = (sh_order+1) * (sh_order+1);

    if (length(normals[pixel_idx]) < 1e-6) {
        output[pixel_idx] = make_float3(0);
        return;
    }

    float2 n_pol = cart2pol(normals[pixel_idx]) / make_float2(CUDART_PI_F, 2*CUDART_PI_F);
    float2 wi_pol = cart2pol(wi_world[pixel_idx]) / make_float2(CUDART_PI_F, 2*CUDART_PI_F);
    float3 result = make_float3(0, 0, 0);
    for (int i = 0; i < max_idx; i++) {
        float bsdf = interp_2d<float>(i, n_pol, bsdf_coeffs);
        float3 envmap = interp_2d<float3>(i, wi_pol, emitter_coeffs);
        result += bsdf * envmap;
    }

    output[pixel_idx] = make_float3(result.x, result.y, result.z);
}

double render_half_lookup(
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> normal,
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> wi_world,
    nb::ndarray<float, nb::ndim<4>, nb::device::cuda, nb::c_contig> bsdf_coeffs,
    nb::ndarray<float, nb::ndim<4>, nb::device::cuda, nb::c_contig> emitter_coeffs,
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> output,
    int sh_order
) {
    // Determine the launch configuration
    // We divide the image into blocks which corrospond to tiles in the image
    int n_rows = normal.shape(0);
    int n_cols = normal.shape(1);
    int n_rows_per_tile = 16;   // Number of rows per tile
    int n_cols_per_tile = 16;   // Number of cols per tile
    int n_tile_rows = n_rows / 16;   // Number of tiles in the row direction
    int n_tile_cols = n_cols / 16;  // Number of tiles in the column direction

    dim3 threads_per_block(n_rows_per_tile, n_cols_per_tile);
    dim3 num_blocks(n_tile_rows, n_tile_cols);

    auto t_start = std::chrono::high_resolution_clock::now();

    dot_product_half_lookup<<<num_blocks, threads_per_block>>>(
        (float3 *)normal.data(), (float3 *)wi_world.data(),             // GBuffer
        (float *)bsdf_coeffs.data(), (float3 *)emitter_coeffs.data(),   // SH textures
        (float3 *)output.data(),                                        // Render buffer
        n_rows, n_cols, sh_order                                        // Metadata
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    double t_elapsed = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    return t_elapsed;
}

// Fast rotation version
__global__
void dot_product_half_fast_rotation(
    float3 *normals, float3 *wi_world, float *alpha,
    float* bsdf_coeffs, float3 *emitter_coeffs,
    float *l_coeffs, float3 *output,
    int n_rows, int n_cols, int n_theta, int sh_order
) {
    // Get the flattened pixel index and 2d position
    uint2 block_start = make_uint2(blockDim.x * blockIdx.x, blockDim.y * blockIdx.y);
    uint2 block_pos = make_uint2(threadIdx.x, threadIdx.y);
    uint2 pixel_pos = block_start + block_pos;
    uint pixel_idx = pixel_pos.x * n_cols + pixel_pos.y;

    // Make sure we are not out of bounds
    if (pixel_idx >= n_rows * n_cols) return;

    int max_idx = (sh_order+1) * (sh_order+1);

    if (length(normals[pixel_idx]) < 1e-6) {
        output[pixel_idx] = make_float3(0);
        return;
    }

    float2 n_pol = cart2pol(normals[pixel_idx]);
    float2 wi_pol = cart2pol(wi_world[pixel_idx]) / make_float2(CUDART_PI_F, 2*CUDART_PI_F);
    float a = alpha[pixel_idx];

    float3 result = make_float3(0, 0, 0);
    for (int i = 0; i < max_idx; i++) {
        float bsdf = fast_rotation(i, n_pol, a, bsdf_coeffs, l_coeffs, n_theta);
        float3 envmap = interp_2d<float3>(i, wi_pol, emitter_coeffs);
        result += bsdf * envmap;
    }

    output[pixel_idx] = result;
}

double render_half_fast_rotation(
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> normals,
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> wi_world,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda, nb::c_contig> alpha,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda, nb::c_contig> bsdf_coeffs,
    nb::ndarray<float, nb::ndim<4>, nb::device::cuda, nb::c_contig> emitter_coeffs,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda, nb::c_contig> l_coeffs,
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> output,
    int sh_order
) {
    assert(sh_order <= MAX_SH_ORDER);

    // Determine the launch configuration
    // We divide the image into blocks which corrospond to tiles in the image
    int n_rows = normals.shape(0);
    int n_cols = normals.shape(1);
    int n_rows_per_tile = 16;   // Number of rows per tile
    int n_cols_per_tile = 16;   // Number of cols per tile
    int n_tile_rows = n_rows / 16;   // Number of tiles in the row direction
    int n_tile_cols = n_cols / 16;  // Number of tiles in the column direction

    dim3 threads_per_block(n_rows_per_tile, n_cols_per_tile);
    dim3 num_blocks(n_tile_rows, n_tile_cols);

    cudaDeviceSynchronize();
#ifndef NDEBUG
    cudaProfilerStart();
#endif

    auto t_start = std::chrono::high_resolution_clock::now();

    dot_product_half_fast_rotation<<<num_blocks, threads_per_block>>>(
        (float3 *)normals.data(), (float3 *)wi_world.data(), (float *)alpha.data(),
        (float *)bsdf_coeffs.data(), (float3 *)emitter_coeffs.data(),
        (float *)l_coeffs.data(), (float3 *)output.data(),
        n_rows, n_cols, l_coeffs.shape(1), sh_order                                              // Metadata
    );
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    double t_elapsed = std::chrono::duration<double, std::milli>(t_end-t_start).count();

#ifndef NDEBUG
    cudaProfilerStop();
#endif

    return t_elapsed;
}

// Texture lookup version
__global__
void dot_product_glint_lookup(
    float3 *onb_s, float3 *onb_t, float3 *onb_n, float3 *wi_world,
    float *bsdf_coeffs, float3 *emitter_coeffs,
    char *active_mask,
    float3 *output,
    int n_rows, int n_cols, int theta_idx, int sh_order, int n_phi
) {
    // Get the flattened pixel index and 2d position
    uint2 block_start = make_uint2(blockDim.x * blockIdx.x, blockDim.y * blockIdx.y);
    uint2 block_pos = make_uint2(threadIdx.x, threadIdx.y);
    uint2 pixel_pos = block_start + block_pos;
    uint pixel_idx = pixel_pos.x * n_cols + pixel_pos.y;
    uint phi_idx = threadIdx.z;

    // Make sure we are not out of bounds
    if (pixel_idx >= n_rows * n_cols) return;
    if (!active_mask[pixel_idx]) return;

    int max_idx = (sh_order+1) * (sh_order+1);
    float3 normal = bin_to_normal(theta_idx, phi_idx, n_phi);
    float nx = onb_s[pixel_idx].x * normal.x + onb_t[pixel_idx].x * normal.y + onb_n[pixel_idx].x * normal.z;
    float ny = onb_s[pixel_idx].y * normal.x + onb_t[pixel_idx].y * normal.y + onb_n[pixel_idx].y * normal.z;
    float nz = onb_s[pixel_idx].z * normal.x + onb_t[pixel_idx].z * normal.y + onb_n[pixel_idx].z * normal.z;
    normal = make_float3(nx, ny, nz);

    float2 n_pol = cart2pol(normal) / make_float2(M_PI, 2*M_PI);
    float2 wi_pol = cart2pol(wi_world[pixel_idx]) / make_float2(M_PI, 2*M_PI);
    float3 result = make_float3(0);
    // Check if the reflected direction is going into the surface.
    if (dot(reflect(wi_world[pixel_idx], normal), onb_n[pixel_idx]) < 0) {
        for (int i = 0; i < max_idx; i++) {
            float bsdf = interp_2d<float>(i, n_pol, bsdf_coeffs);
            float3 envmap = interp_2d<float3>(i, wi_pol, emitter_coeffs);
            result += bsdf * envmap;
        }
    }

    uint output_idx = pixel_pos.x * n_cols * BIN_NPHI + pixel_pos.y * BIN_NPHI + phi_idx;
    output[output_idx] = result;
}

double render_glint_lookup(
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> onb_s,
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> onb_t,
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> onb_n,
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> wi_world,
    nb::ndarray<float, nb::ndim<4>, nb::device::cuda, nb::c_contig> bsdf_coeffs,
    nb::ndarray<float, nb::ndim<4>, nb::device::cuda, nb::c_contig> emitter_coeffs,
    nb::ndarray<char, nb::ndim<2>, nb::device::cuda, nb::c_contig> active_mask,
    nb::ndarray<float, nb::shape<nb::any, nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> output,
    nb::ndarray<float, nb::shape<BIN_NTHETA>, nb::device::cpu, nb::c_contig> theta_bins,
    int theta_idx, int sh_order, int n_phi
) {
    // Determine the launch configuration
    // We divide the image into blocks which correspond to tiles in the image
    int n_rows = onb_n.shape(0);
    int n_cols = onb_n.shape(1);
    int n_rows_per_tile = 4;   // Number of rows per tile
    int n_cols_per_tile = 4;   // Number of cols per tile
    int n_tile_rows = n_rows / n_rows_per_tile;   // Number of tiles in the row direction
    int n_tile_cols = n_cols / n_cols_per_tile;   // Number of tiles in the column direction

    dim3 threads_per_block(n_rows_per_tile, n_cols_per_tile, n_phi);
    dim3 num_blocks(n_tile_rows, n_tile_cols);

    // Copy theta bins to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(g_theta_bins, theta_bins.data(), theta_bins.nbytes(), 0, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

#ifndef NDEBUG
    cudaProfilerStart();
#endif

    auto t_start = std::chrono::high_resolution_clock::now();

    dot_product_glint_lookup<<<num_blocks, threads_per_block>>>(
        (float3 *)onb_s.data(), (float3 *)onb_t.data(), (float3 *)onb_n.data(), (float3 *)wi_world.data(),  // GBuffer
        (float *)bsdf_coeffs.data(), (float3 *)emitter_coeffs.data(),   // SH textures
        (char *)active_mask.data(),
        (float3 *)output.data(),                                        // Render buffer
        n_rows, n_cols, theta_idx, sh_order, n_phi                      // Metadata
    );
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    double t_elapsed = std::chrono::duration<double, std::milli>(t_end-t_start).count();

#ifndef NDEBUG
    cudaProfilerStop();
#endif

    return t_elapsed;
}

// Fast rotation version
__global__
void dot_product_glint_fast_rotation(
    float3 *onb_s, float3 *onb_t, float3 *onb_n, float3 *wi_world, float *alpha,
    float *bsdf_coeffs, float3 *emitter_coeffs, float *l_coeffs,
    char *active_mask,
    float3 *output,
    int n_threads, int theta_idx, int n_theta, int sh_order, int n_phi
) {
    // Get the flattened pixel index and 2d position
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint pixel_idx = idx / n_phi;
    uint phi_idx = idx % n_phi;

    // Make sure we are not out of bounds
    if (idx >= n_threads) return;
    if (!active_mask[pixel_idx]) return;

    int max_idx = (sh_order+1) * (sh_order+1);

    float3 normal = bin_to_normal(theta_idx, phi_idx, n_phi);
    float nx = onb_s[pixel_idx].x * normal.x + onb_t[pixel_idx].x * normal.y + onb_n[pixel_idx].x * normal.z;
    float ny = onb_s[pixel_idx].y * normal.x + onb_t[pixel_idx].y * normal.y + onb_n[pixel_idx].y * normal.z;
    float nz = onb_s[pixel_idx].z * normal.x + onb_t[pixel_idx].z * normal.y + onb_n[pixel_idx].z * normal.z;
    float a = alpha[pixel_idx];
    normal = make_float3(nx, ny, nz);

    float2 n_pol = cart2pol(normal);
    float2 wi_pol = cart2pol(wi_world[pixel_idx]) / make_float2(CUDART_PI_F, 2*CUDART_PI_F);

    float3 result = make_float3(0);
    // Check if the reflected direction is going into the surface.
    if (dot(reflect(wi_world[pixel_idx], normal), onb_n[pixel_idx]) < 0) {
        for (int i = 0; i < max_idx; i++) {
            float bsdf = fast_rotation(i, n_pol, a, bsdf_coeffs, l_coeffs, n_theta);
            float3 envmap = interp_2d<float3>(i, wi_pol, emitter_coeffs);
            result += bsdf * envmap;
        }
    }

    output[idx] = result;
}

double render_glint_fast_rotation(
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> onb_s,
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> onb_t,
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> onb_n,
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> wi_world,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda, nb::c_contig> alpha,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda, nb::c_contig> bsdf_coeffs,
    nb::ndarray<float, nb::ndim<4>, nb::device::cuda, nb::c_contig> emitter_coeffs,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda, nb::c_contig> l_coeffs,
    nb::ndarray<char, nb::ndim<2>, nb::device::cuda, nb::c_contig> active_mask,
    nb::ndarray<float, nb::shape<nb::any, nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> output,
    nb::ndarray<float, nb::ndim<1>, nb::device::cpu, nb::c_contig> theta_bins,
    int theta_idx, int sh_order, int n_phi
) {
    assert(sh_order <= MAX_SH_ORDER);

    // Determine the launch configuration
    int n_rows = onb_n.shape(0);
    int n_cols = onb_n.shape(1);
    int n_threads = n_rows * n_cols * n_phi;
    
    int grid_size, block_size;
    calculate_launch_params(n_threads, &grid_size, &block_size, dot_product_glint_fast_rotation);

    dim3 threads_per_block(block_size, 1, 1);
    dim3 num_blocks(grid_size, 1, 1);

    // Copy theta bins to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(g_theta_bins, theta_bins.data(), theta_bins.nbytes(), 0, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

#ifndef NDEBUG
    cudaProfilerStart();
#endif

    auto t_start = std::chrono::high_resolution_clock::now();

    dot_product_glint_fast_rotation<<<num_blocks, threads_per_block>>>(
        (float3 *)onb_s.data(), (float3 *)onb_t.data(), (float3 *)onb_n.data(), (float3 *)wi_world.data(), (float *)alpha.data(),   // GBuffer
        (float *) bsdf_coeffs.data(), (float3 *)emitter_coeffs.data(), (float *)l_coeffs.data(),                                    // SH textures
        (char *)active_mask.data(),                                                                                                 // Vectorization mask
        (float3 *)output.data(),                                                                                                    // Render buffer
        n_threads, theta_idx, l_coeffs.shape(1), sh_order, n_phi                                                                    // Metadata
    );
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    double t_elapsed = std::chrono::duration<double, std::milli>(t_end-t_start).count();

#ifndef NDEBUG
    cudaProfilerStop();
#endif

    return t_elapsed;
}

NB_MODULE(fast_dot, m) {
    m.def("render_half_lookup", &render_half_lookup);
    m.def("render_half_fast_rotation", &render_half_fast_rotation);
    m.def("render_glint_lookup", &render_glint_lookup);
    m.def("render_glint_fast_rotation", &render_glint_fast_rotation);
}