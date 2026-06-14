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
    OptixTraversableHandle traversable;
};
