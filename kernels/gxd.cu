#include <chrono>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "helper_math.h"
#include "gxd.cuh"
#include "utils.h"

namespace nb = nanobind;

#define M_INV_2_PI 0.15915494309f

#define STACK_DEPTH 52

__device__ float Gr(float2 st1, float2 st2, float alpha) {
    float2 st = st1 - st2;
    // Probably inefficient but will do for now.
    float3 n1 = make_float3(st1.x, st1.y, sqrtf(1-st1.x*st1.x-st1.y*st1.y));
    float3 n2 = make_float3(st2.x, st2.y, sqrtf(1-st2.x*st2.x-st2.y*st2.y));
    float angle = dot(n1, n2);
    float inv_alpha_sq = (1.f / (alpha*alpha));
    float exp = -0.5f * (1 - angle*angle) * inv_alpha_sq ;
    float norm = M_INV_2_PI * inv_alpha_sq;
    return norm * expf(exp);
}

__global__ void D_naive(
    float3 *wh, float2 *cuv, float2 *duv,
    float3 *normalmap, float *output,
    float alpha, float uv_scale,
    int n_threads, int n_res, int grid_size
) {
    // Get the flattened pixel index and 2d position
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we are not out of bounds
    if (idx >= n_threads) return;

    float2 suv = cuv[idx] - (duv[idx] / 2.f);

    float2 Hst = make_float2(wh[idx].x, wh[idx].y);

    float2 uv;
    int idx_x, idx_y;
    float result = 0;
    float count = grid_size * grid_size;
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            uv.x = suv.x + duv[idx].x * ((float)i / grid_size);
            uv.y = suv.y + duv[idx].y * ((float)j / grid_size);
            idx_x = (int)roundf(uv.x * uv_scale * n_res) % n_res;
            idx_y = (int)roundf(uv.y * uv_scale * n_res) % n_res;
            float3 n = normalmap[idx_y * n_res + idx_x];
            n = n * 2 - 1;
            result += Gr(make_float2(n.x, n.y), Hst, alpha);
        }
    }

    output[idx] = result / count;
}

double eval_naive(
    nb::ndarray<float, nb::shape<nb::any, nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> wh,
    nb::ndarray<float, nb::shape<nb::any, nb::any, nb::any, 2>, nb::device::cuda, nb::c_contig> cuv,
    nb::ndarray<float, nb::shape<nb::any, nb::any, nb::any, 2>, nb::device::cuda, nb::c_contig> duv,
    nb::ndarray<float, nb::shape<nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> normalmap,
    nb::ndarray<float, nb::ndim<3>, nb::device::cuda, nb::c_contig> output,
    float alpha, float uv_scale, int grid
) {
    // Determine the launch configuration
    // We divide the image into blocks which corrospond to tiles in the image
    int n_rows = wh.shape(0);
    int n_cols = wh.shape(1);
    int n_spp = wh.shape(2);
    int n_threads = n_rows * n_cols * n_spp;

    int grid_size, block_size;
    calculate_launch_params(n_threads, &grid_size, &block_size, D_naive);

    dim3 threads_per_block(block_size, 1, 1);
    dim3 num_blocks(grid_size, 1, 1);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Launch and check for error
    D_naive<<<num_blocks, threads_per_block>>>(
        (float3 *)wh.data(), (float2 *)cuv.data(), (float2 *)duv.data(),
        (float3 *)normalmap.data(), (float *)output.data(),
        alpha, uv_scale,
        n_threads, normalmap.shape(0), grid
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    double t_elapsed = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    return t_elapsed;
}

__device__ float2 prune(
    float2 suv, float2 euv, float2 wh_st, float4 *mipmap,
    float alpha, int n_res, int n_miplevels
) {
    // Get integer coordinates of pixel footprint corners
    int isu = floorf(suv.y * n_res);
    int isv = floorf(suv.x * n_res);
    int ieu = ceilf(euv.y * n_res);
    int iev = ceilf(euv.x * n_res);
    if (ieu >= n_res) ieu -= 1;
    if (iev >= n_res) iev -= 1;

    // Get number of texels in the footprint
    float count = (ieu-isu+1)*(iev-isv+1);

    // Get bounds in direction
    float slo = wh_st.x - 5*alpha;
    float shi = wh_st.x + 5*alpha;
    float tlo = wh_st.y - 5*alpha;
    float thi = wh_st.y + 5*alpha;

    // Initialize a stack to maintain variables
    int mip_idx[STACK_DEPTH];
    int mip_i[STACK_DEPTH];
    int mip_j[STACK_DEPTH];
    int stack_ptr = 0;
    mip_idx[0] = n_miplevels - 1;
    mip_i[0] = 0;
    mip_j[0] = 0;

    // Start hierarchy traversal
    float result = 0.0;
    do {
        int cur_mip_idx = mip_idx[stack_ptr];
        int cur_mip_i = mip_i[stack_ptr];
        int cur_mip_j = mip_j[stack_ptr];

        int mip_level_offset = cur_mip_idx * n_res * n_res;
        float4 st = mipmap[mip_level_offset + cur_mip_i * n_res + cur_mip_j];

        bool intersect = true;
        intersect = intersect && range_intersect(isu, ieu, cur_mip_i << cur_mip_idx, ((cur_mip_i + 1) << cur_mip_idx) - 1);
        intersect = intersect && range_intersect(isv, iev, cur_mip_j << cur_mip_idx, ((cur_mip_j + 1) << cur_mip_idx) - 1);
        intersect = intersect && range_intersect(slo, shi, st.x, st.z);
        intersect = intersect && range_intersect(tlo, thi, st.y, st.w);
        if (!intersect) {
            stack_ptr--;
            continue;
        }

        if (cur_mip_idx == 0) {
            result += Gr(wh_st, make_float2(st.x, st.y), alpha);
            stack_ptr--;
            continue;
        }

        // Add 4 children
        // We can replace the current node
        mip_idx[stack_ptr] = cur_mip_idx-1;
        mip_i[stack_ptr] = cur_mip_i * 2;
        mip_j[stack_ptr] = cur_mip_j * 2;

        stack_ptr++;
        mip_idx[stack_ptr] = cur_mip_idx-1;
        mip_i[stack_ptr] = cur_mip_i * 2 + 1;
        mip_j[stack_ptr] = cur_mip_j * 2;

        stack_ptr++;
        mip_idx[stack_ptr] = cur_mip_idx-1;
        mip_i[stack_ptr] = cur_mip_i * 2;
        mip_j[stack_ptr] = cur_mip_j * 2 + 1;

        stack_ptr++;
        mip_idx[stack_ptr] = cur_mip_idx-1;
        mip_i[stack_ptr] = cur_mip_i * 2 + 1;
        mip_j[stack_ptr] = cur_mip_j * 2 + 1;
    } while (stack_ptr >= 0);

    return make_float2(result, count);
}

__global__ void D_accel(
    float3 *wh, float2 *cuv, float2 *duv, char *active_mask,
    float4 *mipmap, float *output,
    float alpha,
    int n_threads, int n_res, int n_miplevels
) {
    // Get the flattened pixel index and 2d position
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we are not out of bounds
    if (idx >= n_threads || !active_mask[idx]) return;

    // Compute footprint bounds
    float2 suv = cuv[idx] - (duv[idx] / 2.f);
    float2 euv = cuv[idx] + (duv[idx] / 2.f);
    float2 fsuv = make_float2(floorf(suv.x), floorf(suv.y));
    float2 ceuv = make_float2(ceilf(euv.x), ceilf(euv.y));

    // Take care of tiling
    float2 result = make_float2(0, 0);
    for(int i = fsuv.x; i <= ceuv.x; i++) {
        float su = suv.x - i;
        float eu = euv.x - i;
        su = su < 0.f ? 0.f : su;
        eu = eu > 1.f ? 1.f : eu;
        if (su > eu) continue;

        for(int j = fsuv.y; j <= ceuv.y; j++) {
            float sv = suv.y - j;
            float ev = euv.y - j;
            sv = sv < 0.f ? 0.f : sv;
            ev = ev > 1.f ? 1.f : ev;
            if (sv > ev) continue;

            result += prune(
                make_float2(su, sv), make_float2(eu, ev),
                make_float2(wh[idx].x, wh[idx].y), mipmap,
                alpha, n_res, n_miplevels
            );
        }
    }

    if (result.x < 0 || isnan(result.x)) output[idx] = 0.0;
    else output[idx] = result.x / result.y;
}

// TODO: Add a mask variable
double eval_accel(
    nb::ndarray<float, nb::shape<nb::any, nb::any, nb::any, 3>, nb::device::cuda, nb::c_contig> wh,
    nb::ndarray<float, nb::shape<nb::any, nb::any, nb::any, 2>, nb::device::cuda, nb::c_contig> cuv,
    nb::ndarray<float, nb::shape<nb::any, nb::any, nb::any, 2>, nb::device::cuda, nb::c_contig> duv,
    nb::ndarray<char, nb::ndim<3>, nb::device::cuda, nb::c_contig> active_mask,
    nb::ndarray<float, nb::shape<nb::any, nb::any, nb::any, 4>, nb::device::cuda, nb::c_contig> mipmap,
    nb::ndarray<float, nb::ndim<3>, nb::device::cuda, nb::c_contig> output,
    float alpha
) {
    // Determine the launch configuration
    int n_rows = wh.shape(0);
    int n_cols = wh.shape(1);
    int n_spp = wh.shape(2);
    int n_threads = n_rows * n_cols * n_spp;

    int grid_size, block_size;
    calculate_launch_params(n_threads, &grid_size, &block_size, D_accel);

    dim3 threads_per_block(block_size, 1, 1);
    dim3 num_blocks(grid_size, 1, 1);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Launch and check for error
    D_accel<<<num_blocks, threads_per_block>>>(
        (float3 *)wh.data(), (float2 *)cuv.data(), (float2 *)duv.data(), (char *)active_mask.data(),
        (float4 *)mipmap.data(), (float *)output.data(),
        alpha, n_threads, mipmap.shape(1), mipmap.shape(0)
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    double t_elapsed = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    return t_elapsed;
}

NB_MODULE(gxd, m) {
    m.def("eval_naive", &eval_naive);
    m.def("eval_accel", &eval_accel);
}