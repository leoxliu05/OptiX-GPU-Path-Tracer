#pragma once

#include <cuda_runtime.h>

#include <filesystem>
#include <vector>

struct objectTransform
{
    float3 scale = make_float3(1.0f, 1.0f, 1.0f);
    float3 translation = make_float3(0.0f, 0.0f, 0.0f);
};

std::vector<float3> loadObjTriangles(
    const std::filesystem::path& path,
    const objectTransform& transform
);

