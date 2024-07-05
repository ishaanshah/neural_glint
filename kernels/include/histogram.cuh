#pragma once

#include "common.h"
#include "transforms.cuh"
#include "math_constants.h"

__device__
float3 bin_to_normal(uint theta_idx, uint phi_idx, int n_phi) {
    float dphi = 1.f / float(n_phi);

    float theta = g_theta_bins[theta_idx];
    float phi = ((float(phi_idx) * dphi) + (dphi / 2)) * 2*CUDART_PI_F;

    return pol2cart(make_float2(theta, phi));
}