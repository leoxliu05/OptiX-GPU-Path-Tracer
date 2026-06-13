#include <cuda_runtime.h>
#include <optix_device.h>

#include "LaunchParams.h"

extern "C" {
__constant__ LaunchParams params;
}

static __forceinline__ __device__ unsigned char toByte(float value)
{
    value = fminf(fmaxf(value, 0.0f), 1.0f);
    return static_cast<unsigned char>(255.99f * sqrtf(value));
}

static __forceinline__ __device__ float3 normalize3(float3 value)
{
    const float invLength = rsqrtf(
        value.x * value.x + value.y * value.y + value.z * value.z
    );
    return make_float3(
        value.x * invLength,
        value.y * invLength,
        value.z * invLength
    );
}

extern "C" __global__ void __raygen__render()
{
    const uint3 index = optixGetLaunchIndex();
    const float2 pixel = make_float2(
        (static_cast<float>(index.x) + 0.5f) / params.width,
        (static_cast<float>(index.y) + 0.5f) / params.height
    );

    const float aspect = static_cast<float>(params.width) / params.height;
    const float screenX = (2.0f * pixel.x - 1.0f) * aspect;
    const float screenY = 1.0f - 2.0f * pixel.y;

    const float3 origin = make_float3(0.0f, 0.0f, 2.0f);
    const float3 direction = normalize3(make_float3(screenX, screenY, -1.5f));

    unsigned int red = __float_as_uint(0.0f);
    unsigned int green = __float_as_uint(0.0f);
    unsigned int blue = __float_as_uint(0.0f);

    optixTrace(
        params.traversable,
        origin,
        direction,
        0.001f,
        1.0e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0,
        red,
        green,
        blue
    );

    const float3 color = make_float3(
        __uint_as_float(red),
        __uint_as_float(green),
        __uint_as_float(blue)
    );

    params.image[index.y * params.width + index.x] = make_uchar4(
        toByte(color.x),
        toByte(color.y),
        toByte(color.z),
        255
    );
}

extern "C" __global__ void __miss__radiance()
{
    const float3 direction = optixGetWorldRayDirection();
    const float t = 0.5f * (direction.y + 1.0f);
    const float3 color = make_float3(
        (1.0f - t) + 0.45f * t,
        (1.0f - t) + 0.65f * t,
        (1.0f - t) + 1.00f * t
    );

    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

extern "C" __global__ void __closesthit__radiance()
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float u = barycentrics.x;
    const float v = barycentrics.y;
    const float w = 1.0f - u - v;

    optixSetPayload_0(__float_as_uint(0.08f + 0.92f * w));
    optixSetPayload_1(__float_as_uint(0.08f + 0.92f * u));
    optixSetPayload_2(__float_as_uint(0.08f + 0.92f * v));
}

