#include <cuda_runtime.h>
#include <optix_device.h>

#include "optix_path_tracer/shared_types.h"

// OptiX binds this constant-memory symbol to the launch parameter buffer named
// by pipelineLaunchParamsVariableName on the host.
extern "C" {
__constant__ launchParams params;
}

namespace {

constexpr float pi = 3.14159265358979323846f;

// A trace writes its closest-hit result into this caller-owned payload. The
// pointer is transported through two 32-bit OptiX payload registers.
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

static __forceinline__ __device__ float3 zeroVector()
{
    return make_float3(0.0f, 0.0f, 0.0f);
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

// Lightweight per-pixel pseudo-random number generator. It is sufficient for
// this baseline renderer but can later be replaced by a higher-quality sampler.
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
    // Build an orthonormal basis around the shading normal. Selecting a helper
    // axis that is not parallel to the normal avoids a degenerate cross product.
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
    // Mapping a uniform square sample to a cosine-weighted hemisphere matches
    // the Lambertian BRDF, so its BRDF and PDF terms cancel in throughput.
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
    // OptiX payload registers are 32-bit, while device pointers are 64-bit.
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

    // Ray type zero is radiance. A stride of two skips the interleaved shadow
    // record for each material.
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
        2,
        0,
        firstPayload,
        secondPayload
    );
}

static __forceinline__ __device__ bool isVisible(
    float3 origin,
    float3 direction,
    float maximumDistance
)
{
    unsigned int visible = 1;

    // Ray type one uses a minimal any-hit program. It clears the visibility
    // payload and terminates as soon as any blocker is found.
    optixTrace(
        params.traversable,
        origin,
        direction,
        0.001f,
        maximumDistance,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        1,
        2,
        1,
        visible
    );
    return visible != 0;
}

static __forceinline__ __device__ const areaLightData& selectAreaLight(
    unsigned int& randomState
)
{
    // Choosing a triangle in proportion to its area, followed by uniform point
    // sampling on that triangle, is equivalent to sampling the union of all
    // emissive triangles uniformly by area.
    const float areaSample = randomFloat(randomState) * params.totalLightArea;
    float accumulatedArea = 0.0f;
    for (unsigned int index = 0; index < params.lightCount; ++index) {
        accumulatedArea += params.lights[index].area;
        if (areaSample <= accumulatedArea) {
            return params.lights[index];
        }
    }
    return params.lights[params.lightCount - 1];
}

static __forceinline__ __device__ float3 sampleTrianglePoint(
    const areaLightData& light,
    unsigned int& randomState
)
{
    // The square root produces uniform area density over the triangle.
    const float rootSample = sqrtf(randomFloat(randomState));
    const float secondSample = randomFloat(randomState);
    const float firstWeight = 1.0f - rootSample;
    const float secondWeight = rootSample * (1.0f - secondSample);
    const float thirdWeight = rootSample * secondSample;
    return add(
        add(
            scale(light.firstVertex, firstWeight),
            scale(light.secondVertex, secondWeight)
        ),
        scale(light.thirdVertex, thirdWeight)
    );
}

static __forceinline__ __device__ float3 estimateDirectLighting(
    float3 position,
    float3 normal,
    float3 albedo,
    unsigned int& randomState
)
{
    if (params.lightCount == 0 || params.totalLightArea <= 0.0f) {
        return zeroVector();
    }

    const areaLightData& light = selectAreaLight(randomState);
    const float3 lightPosition = sampleTrianglePoint(light, randomState);
    const float3 toLight = subtract(lightPosition, position);
    const float distanceSquared = dot(toLight, toLight);
    if (distanceSquared <= 1.0e-8f) {
        return zeroVector();
    }

    const float distance = sqrtf(distanceSquared);
    if (distance <= 0.003f) {
        return zeroVector();
    }
    const float3 lightDirection = scale(toLight, 1.0f / distance);
    const float surfaceCosine = fmaxf(dot(normal, lightDirection), 0.0f);
    const float lightCosine = fmaxf(dot(light.normal, scale(lightDirection, -1.0f)), 0.0f);
    if (surfaceCosine <= 0.0f || lightCosine <= 0.0f) {
        return zeroVector();
    }

    const float3 shadowOrigin = add(position, scale(normal, 0.001f));
    if (!isVisible(shadowOrigin, lightDirection, distance - 0.002f)) {
        return zeroVector();
    }

    // The requested estimator uses a uniform area PDF. Lambertian BRDF is
    // albedo / pi, and the two cosine terms plus inverse-square distance convert
    // emitted radiance at the sampled light point into incident contribution.
    const float lightPdf = 1.0f / params.totalLightArea;
    const float geometryTerm =
        surfaceCosine * lightCosine / distanceSquared;
    return scale(
        multiply(albedo, light.emission),
        geometryTerm / (pi * lightPdf)
    );
}

static __forceinline__ __device__ float3 toneMap(float3 color)
{
    // ACES-style fitted curve compresses HDR radiance into display range.
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
    // OptiX launches one ray-generation invocation for each output pixel.
    const uint3 launchIndex = optixGetLaunchIndex();
    const unsigned int pixelIndex =
        launchIndex.y * params.width + launchIndex.x;
    unsigned int randomState = hash(pixelIndex ^ params.seed);
    float3 accumulatedRadiance = make_float3(0.0f, 0.0f, 0.0f);

    for (unsigned int sampleIndex = 0;
         sampleIndex < params.samplesPerPixel;
         ++sampleIndex) {
        // Jitter the sample inside the pixel to perform stochastic anti-aliasing.
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
        float3 directRadiance = zeroVector();
        float3 indirectRadiance = zeroVector();

        // Each loop iteration traces one path segment. Keeping the loop here
        // means the OptiX pipeline itself only requires a trace depth of one.
        for (unsigned int depth = 0; depth < params.maxDepth; ++depth) {
            pathPayload payload{};
            traceRay(rayOrigin, rayDirection, payload);

            if (!payload.hit) {
                // A miss terminates the path with environment radiance.
                const float3 environmentContribution = multiply(
                    throughput,
                    params.environmentColor
                );
                if (depth == 0) {
                    directRadiance = add(
                        directRadiance,
                        environmentContribution
                    );
                } else {
                    indirectRadiance = add(
                        indirectRadiance,
                        environmentContribution
                    );
                }
                break;
            }

            if (largestComponent(payload.material.emission) > 0.0f) {
                // Without MIS, only camera-visible emission is accumulated here.
                // Diffuse paths reaching a light are already represented by the
                // explicit area-light sample taken at the previous surface.
                if (depth == 0) {
                    directRadiance = add(
                        directRadiance,
                        multiply(throughput, payload.material.emission)
                    );
                }
                break;
            }

            float3 surfaceNormal = payload.normal;
            if (dot(surfaceNormal, rayDirection) > 0.0f) {
                surfaceNormal = scale(surfaceNormal, -1.0f);
            }

            const float3 sampledDirectLight = multiply(
                throughput,
                estimateDirectLighting(
                    payload.position,
                    surfaceNormal,
                    payload.material.albedo,
                    randomState
                )
            );
            if (depth == 0) {
                directRadiance = add(directRadiance, sampledDirectLight);
            } else {
                indirectRadiance = add(indirectRadiance, sampledDirectLight);
            }

            throughput = multiply(throughput, payload.material.albedo);

            // Offset the next origin to avoid immediately hitting the same
            // triangle because of floating-point intersection error.
            rayOrigin = add(payload.position, scale(surfaceNormal, 0.001f));
            rayDirection = sampleCosineHemisphere(surfaceNormal, randomState);

            if (depth >= 3) {
                // Russian roulette terminates low-throughput paths without
                // bias. Surviving paths are reweighted by the inverse chance.
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

        accumulatedRadiance = add(
            accumulatedRadiance,
            add(directRadiance, indirectRadiance)
        );
    }

    const float inverseSampleCount = 1.0f / params.samplesPerPixel;

    // Average all Monte Carlo samples before tone mapping and gamma conversion.
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
    // The ray-generation loop handles environment lighting after observing this
    // flag, which keeps the miss program independent of integrator policy.
    pathPayload* payload = reinterpret_cast<pathPayload*>(unpackPointer(
        optixGetPayload_0(),
        optixGetPayload_1()
    ));
    payload->hit = false;
}

extern "C" __global__ void __miss__shadow()
{
    // The visibility payload starts at one and remains unchanged on a miss.
}

extern "C" __global__ void __anyhit__shadow()
{
    optixSetPayload_0(0);
    optixTerminateRay();
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

    // Primitive indices address the flat, non-indexed triangle array uploaded
    // by the host. Each primitive therefore occupies three consecutive vertices.
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

    // Geometry normals are sufficient for the current flat-shaded OBJ scenes.
    payload->normal = normalize(cross(
        subtract(secondVertex, firstVertex),
        subtract(thirdVertex, firstVertex)
    ));
    payload->material = groupData->material;
}
