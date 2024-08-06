#pragma once
#include "helper_math.h"
#include "common.h"
#include "math_constants.h"

#define M_SQRT_TWO 1.41421356237f

__device__
int2 get_lm(uint sh_idx) {
    int l = sqrtf(float(sh_idx));
    int m = sh_idx - l*l - l;
    return make_int2(l, m);
}

__device__ __forceinline__
uint flat_idx(uint sh_idx, uint theta_idx, uint phi_idx) {
    return sh_idx * SH_STRIDE + theta_idx * SH_NPHI + phi_idx;
}

__device__ __forceinline__
uint wrap_idx(int idx, int max_idx) {
    return (idx + max_idx) % max_idx;
}

template <typename T>
__device__ T interp_1d(float f_idx, T* data, uint size) {
    f_idx -= floorf(f_idx);  // Wrap to 0-1
    f_idx *= size;
    int idx_lo = floor(f_idx);
    float t = f_idx - idx_lo;
    T result = data[idx_lo] * (1-t) + data[idx_lo+1] * t;

    return result;
}

// WARNING: Wrapping not supported
template <typename T>
__device__ T interp_2d(uint sh_idx, float2 f_idx, T* data) {
    // Wrap texture coordinates to 0 -> 1
    f_idx -= floorf(f_idx);

    // Convert from normalized to texture resolution (-0.5 -> res-0.5)
    f_idx.x = f_idx.x * SH_NTHETA - 0.5f;
    f_idx.y = f_idx.y * SH_NPHI - 0.5f;
    // f_idx.x = f_idx.x * SH_NTHETA;
    // f_idx.y = f_idx.y * SH_NPHI;
    
    // Get integer coordinate of lowest corner (-1 -> res-1)
    int2 idx = make_int2((int)floorf(f_idx.x), (int)floorf(f_idx.y));

    // Get distance from the lowest corner (0 -> 1)
    float2 w = f_idx - floorf(f_idx);

    T result = {0};
    result += w.x * w.y * data[flat_idx(sh_idx, wrap_idx(idx.x+1, SH_NTHETA), wrap_idx(idx.y + 1, SH_NPHI))];
    result += (1-w.x) * w.y * data[flat_idx(sh_idx, wrap_idx(idx.x, SH_NTHETA), wrap_idx(idx.y + 1, SH_NPHI))];
    result += w.x * (1-w.y) * data[flat_idx(sh_idx, wrap_idx(idx.x + 1, SH_NTHETA), wrap_idx(idx.y, SH_NPHI))];
    result += (1-w.x) * (1-w.y) * data[flat_idx(sh_idx, wrap_idx(idx.x, SH_NTHETA), wrap_idx(idx.y, SH_NPHI))];

    // Corner case at the poles
    if (idx.x == -1 || idx.x == SH_NTHETA-1) {
        float theta_idx;
        if (idx.x == -1) theta_idx = 0;
        else theta_idx = idx.x;

        result = w.y * data[flat_idx(sh_idx, theta_idx, wrap_idx(idx.y + 1, SH_NPHI))] + \
                 (1-w.y) * data[flat_idx(sh_idx, theta_idx, wrap_idx(idx.y, SH_NPHI))];
    }

    return result;
}

__device__
float eval_sh(uint sh_idx, float2 pol, float* l_coeffs, uint n_theta) {
    int m = get_lm(sh_idx).y;
    float result = interp_1d<float>(pol.x / CUDART_PI_F, &l_coeffs[sh_idx*n_theta], n_theta);

    float angle = abs(m) * pol.y;
    if (m > 0) result *= cosf(angle) * M_SQRT_TWO;
    else if (m < 0) result *= sinf(angle) * M_SQRT_TWO;

    return result;
}

__device__
float fast_rotation(uint sh_idx, float2 pol, float alpha, float *bsdf_coeffs, float* l_coeffs, uint n_theta) {
    int l = get_lm(sh_idx).x;

    // TODO: Remove hardcoded values
    // Find nearest alpha values to interpolate from
    alpha = ((alpha - 0.01f) / 0.98f) * 1000.0f;
    int lo = min((int)alpha, 999);
    int hi = min(lo + 1, 999);

    // Interpolate to get SH coefficient for given alpha value
    float w = alpha - lo;
    float coeff = w * bsdf_coeffs[l*1000+lo] + (1-w)*bsdf_coeffs[l*1000+hi];
    if (lo == hi)
        coeff = bsdf_coeffs[l*1000+lo];

    // Rotate the given coeff
    float nl = sqrtf(CUDART_PI_F * 4 / (2*l + 1));
    return nl * eval_sh(sh_idx, pol, l_coeffs, n_theta) * coeff;
}