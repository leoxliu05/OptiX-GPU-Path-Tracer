#pragma once

#include <cuda_runtime.h>

#include <cmath>

// Small host-side float3 helpers used to construct the camera basis.
inline float3 subtract(float3 first, float3 second)
{
    return make_float3(
        first.x - second.x,
        first.y - second.y,
        first.z - second.z
    );
}

inline float3 multiply(float3 value, float factor)
{
    return make_float3(value.x * factor, value.y * factor, value.z * factor);
}

inline float dot(float3 first, float3 second)
{
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

inline float3 cross(float3 first, float3 second)
{
    return make_float3(
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x
    );
}

inline float3 normalize(float3 value)
{
    return multiply(value, 1.0f / std::sqrt(dot(value, value)));
}
