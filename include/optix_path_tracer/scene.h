#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "optix_path_tracer/shared_types.h"

// Rendering controls read from the renderer section of scene.json.
struct renderSettings
{
    unsigned int width = 512;
    unsigned int height = 512;
    unsigned int samplesPerPixel = 64;
    unsigned int maxDepth = 8;
    unsigned int seed = 1;
    std::filesystem::path outputPath = "render.ppm";
};

// Pinhole camera parameters expressed in world space.
struct cameraSettings
{
    float3 position{};
    float3 target{};
    float3 up = make_float3(0.0f, 1.0f, 0.0f);
    float verticalFovDegrees = 45.0f;
};

// Fully loaded scene consumed by the renderer. OBJ meshes are flattened into
// triangles, and each triangle has a matching material index.
struct sceneData
{
    renderSettings renderer;
    cameraSettings camera;
    float3 environmentColor{};
    std::vector<float3> vertices;
    std::vector<std::uint32_t> materialIndices;
    std::vector<materialData> materials;
};
