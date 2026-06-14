#pragma once

#include <cuda_runtime.h>
#include <optix.h>

struct materialData
{
    float3 albedo;
    float3 emission;
};

struct hitGroupData
{
    const float3* vertices;
    materialData material;
};

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

