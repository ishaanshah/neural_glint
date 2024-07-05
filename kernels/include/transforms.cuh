#pragma once
#include "helper_math.h"

__device__
float2 cart2pol(float3 dir) {
    float theta = acosf(clamp(dir.z, -1.0f, 1.0f));
    float phi = atan2f(dir.y, dir.x);
    if (phi < 0) phi += 2*CUDART_PI_F;

    return make_float2(theta, phi);
}

__device__
float3 pol2cart(float2 pol) {
    float sin_theta, cos_theta, sin_phi, cos_phi;
    sincosf(pol.x, &sin_theta, &cos_theta);
    sincosf(pol.y, &sin_phi, &cos_phi);

    return make_float3(sin_theta*cos_phi, sin_theta*sin_phi, cos_theta);
}