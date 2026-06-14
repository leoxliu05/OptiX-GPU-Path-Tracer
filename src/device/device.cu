#include <cuda_runtime.h>
#include <optix_device.h>

#include "optix_path_tracer/shared_types.h"

extern "C" {
__constant__ launchParams params;
}

namespace {

constexpr float pi = 3.14159265358979323846f;

struct pathPayload
{
    bool hit;
    float3 position;
    float3 normal;
    materialData material;
};

static __forceinline__ __device__ float3 add(float3 first, float3 second)
{
    return make_float3(
        first.x + second.x,
        first.y + second.y,
        first.z + second.z
    );
}

static __forceinline__ __device__ float3 subtract(float3 first, float3 second)
{
    return make_float3(
        first.x - second.x,
        first.y - second.y,
        first.z - second.z
    );
}

static __forceinline__ __device__ float3 multiply(float3 first, float3 second)
{
    return make_float3(
        first.x * second.x,
        first.y * second.y,
        first.z * second.z
    );
}

static __forceinline__ __device__ float3 scale(float3 value, float factor)
{
    return make_float3(value.x * factor, value.y * factor, value.z * factor);
}

static __forceinline__ __device__ float dot(float3 first, float3 second)
{
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

static __forceinline__ __device__ float3 cross(float3 first, float3 second)
{
    return make_float3(
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x
    );
}

static __forceinline__ __device__ float3 normalize(float3 value)
{
    return scale(value, rsqrtf(fmaxf(dot(value, value), 1.0e-20f)));
}

static __forceinline__ __device__ float largestComponent(float3 value)
{
    return fmaxf(value.x, fmaxf(value.y, value.z));
}

static __forceinline__ __device__ unsigned int hash(unsigned int value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

static __forceinline__ __device__ float randomFloat(unsigned int& state)
{
    state = 1664525u * state + 1013904223u;
    return static_cast<float>(state & 0x00ffffffu) / 16777216.0f;
}

static __forceinline__ __device__ float3 localToWorld(
    float3 localDirection,
    float3 normal
)
{
    const float3 helper = fabsf(normal.z) < 0.999f
        ? make_float3(0.0f, 0.0f, 1.0f)
        : make_float3(1.0f, 0.0f, 0.0f);
    const float3 tangent = normalize(cross(helper, normal));
    const float3 bitangent = cross(normal, tangent);
    return normalize(add(
        add(
            scale(tangent, localDirection.x),
            scale(bitangent, localDirection.y)
        ),
        scale(normal, localDirection.z)
    ));
}

static __forceinline__ __device__ float3 sampleCosineHemisphere(
    float3 normal,
    unsigned int& randomState
)
{
    const float firstSample = randomFloat(randomState);
    const float secondSample = randomFloat(randomState);
    const float radius = sqrtf(firstSample);
    const float angle = 2.0f * pi * secondSample;
    return localToWorld(
        make_float3(
            radius * cosf(angle),
            radius * sinf(angle),
            sqrtf(1.0f - firstSample)
        ),
        normal
    );
}

static __forceinline__ __device__ void packPointer(
    void* pointer,
    unsigned int& firstPayload,
    unsigned int& secondPayload
)
{
    const unsigned long long pointerValue =
        reinterpret_cast<unsigned long long>(pointer);
    firstPayload = static_cast<unsigned int>(pointerValue);
    secondPayload = static_cast<unsigned int>(pointerValue >> 32);
}

static __forceinline__ __device__ void* unpackPointer(
    unsigned int firstPayload,
    unsigned int secondPayload
)
{
    const unsigned long long pointerValue =
        static_cast<unsigned long long>(firstPayload) |
        (static_cast<unsigned long long>(secondPayload) << 32);
    return reinterpret_cast<void*>(pointerValue);
}

static __forceinline__ __device__ void traceRay(
    float3 origin,
    float3 direction,
    pathPayload& payload
)
{
    unsigned int firstPayload;
    unsigned int secondPayload;
    packPointer(&payload, firstPayload, secondPayload);
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
        firstPayload,
        secondPayload
    );
}

static __forceinline__ __device__ float3 toneMap(float3 color)
{
    const float first = 2.51f;
    const float second = 0.03f;
    const float third = 2.43f;
    const float fourth = 0.59f;
    const float fifth = 0.14f;
    return make_float3(
        color.x * (first * color.x + second) /
            (color.x * (third * color.x + fourth) + fifth),
        color.y * (first * color.y + second) /
            (color.y * (third * color.y + fourth) + fifth),
        color.z * (first * color.z + second) /
            (color.z * (third * color.z + fourth) + fifth)
    );
}

static __forceinline__ __device__ unsigned char toByte(float value)
{
    const float gammaCorrected = sqrtf(fminf(fmaxf(value, 0.0f), 1.0f));
    return static_cast<unsigned char>(255.99f * gammaCorrected);
}

} // namespace

extern "C" __global__ void __raygen__render()
{
    const uint3 launchIndex = optixGetLaunchIndex();
    const unsigned int pixelIndex =
        launchIndex.y * params.width + launchIndex.x;
    unsigned int randomState = hash(pixelIndex ^ params.seed);
    float3 accumulatedRadiance = make_float3(0.0f, 0.0f, 0.0f);

    for (unsigned int sampleIndex = 0;
         sampleIndex < params.samplesPerPixel;
         ++sampleIndex) {
        const float screenX =
            2.0f * (static_cast<float>(launchIndex.x) + randomFloat(randomState)) /
                params.width -
            1.0f;
        const float screenY =
            1.0f -
            2.0f * (static_cast<float>(launchIndex.y) + randomFloat(randomState)) /
                params.height;

        float3 rayOrigin = params.cameraOrigin;
        float3 rayDirection = normalize(add(
            add(params.cameraW, scale(params.cameraU, screenX)),
            scale(params.cameraV, screenY)
        ));
        float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
        float3 radiance = make_float3(0.0f, 0.0f, 0.0f);

        for (unsigned int depth = 0; depth < params.maxDepth; ++depth) {
            pathPayload payload{};
            traceRay(rayOrigin, rayDirection, payload);

            if (!payload.hit) {
                radiance = add(
                    radiance,
                    multiply(throughput, params.environmentColor)
                );
                break;
            }

            if (largestComponent(payload.material.emission) > 0.0f) {
                radiance = add(
                    radiance,
                    multiply(throughput, payload.material.emission)
                );
                break;
            }

            float3 surfaceNormal = payload.normal;
            if (dot(surfaceNormal, rayDirection) > 0.0f) {
                surfaceNormal = scale(surfaceNormal, -1.0f);
            }

            throughput = multiply(throughput, payload.material.albedo);
            rayOrigin = add(payload.position, scale(surfaceNormal, 0.001f));
            rayDirection = sampleCosineHemisphere(surfaceNormal, randomState);

            if (depth >= 3) {
                const float survivalProbability = fminf(
                    largestComponent(throughput),
                    0.95f
                );
                if (randomFloat(randomState) > survivalProbability) {
                    break;
                }
                throughput = scale(
                    throughput,
                    1.0f / fmaxf(survivalProbability, 0.05f)
                );
            }
        }

        accumulatedRadiance = add(accumulatedRadiance, radiance);
    }

    const float inverseSampleCount = 1.0f / params.samplesPerPixel;
    const float3 finalColor = toneMap(scale(
        accumulatedRadiance,
        inverseSampleCount
    ));
    params.image[pixelIndex] = make_uchar4(
        toByte(finalColor.x),
        toByte(finalColor.y),
        toByte(finalColor.z),
        255
    );
}

extern "C" __global__ void __miss__radiance()
{
    pathPayload* payload = reinterpret_cast<pathPayload*>(unpackPointer(
        optixGetPayload_0(),
        optixGetPayload_1()
    ));
    payload->hit = false;
}

extern "C" __global__ void __closesthit__radiance()
{
    pathPayload* payload = reinterpret_cast<pathPayload*>(unpackPointer(
        optixGetPayload_0(),
        optixGetPayload_1()
    ));
    const hitGroupData* groupData = reinterpret_cast<const hitGroupData*>(
        optixGetSbtDataPointer()
    );
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    const float3 firstVertex = groupData->vertices[primitiveIndex * 3 + 0];
    const float3 secondVertex = groupData->vertices[primitiveIndex * 3 + 1];
    const float3 thirdVertex = groupData->vertices[primitiveIndex * 3 + 2];
    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDirection = optixGetWorldRayDirection();

    payload->hit = true;
    payload->position = add(
        rayOrigin,
        scale(rayDirection, optixGetRayTmax())
    );
    payload->normal = normalize(cross(
        subtract(secondVertex, firstVertex),
        subtract(thirdVertex, firstVertex)
    ));
    payload->material = groupData->material;
}
