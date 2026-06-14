#pragma once

#include <cuda_runtime.h>
#include <optix.h>

// This header is compiled by both the host compiler and NVCC. Keep these
// structures free of host-only types such as std::vector or std::filesystem.

// Diffuse and emissive properties shared by host SBT setup and device code.
struct materialData
{
    float3 albedo;
    float3 emission;
};

// One emissive triangle sampled uniformly in area for direct lighting.
struct areaLightData
{
    float3 firstVertex;
    float3 secondVertex;
    float3 thirdVertex;
    float3 normal;
    float3 emission;
    float area;
};

// Per-material data attached to an OptiX hit-group SBT record.
struct hitGroupData
{
    const float3* vertices;
    materialData material;
};

// Read-only parameters copied to the device before each OptiX launch.
struct launchParams
{
    uchar4* image;
    unsigned int width;
    unsigned int height;
    unsigned int samplesPerPixel;
    unsigned int maxDepth;
    unsigned int seed;
    float3 cameraOrigin;
    float3 cameraU;
    float3 cameraV;
    float3 cameraW;
    float3 environmentColor;
    const areaLightData* lights;
    unsigned int lightCount;
    float totalLightArea;
    OptixTraversableHandle traversable;
};
