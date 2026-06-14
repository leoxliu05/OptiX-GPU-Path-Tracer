#pragma once

#include <cuda_runtime.h>

#include <filesystem>
#include <vector>

#include "optix_path_tracer/scene.h"

// Host-side image returned after the OptiX launch finishes.
struct renderResult
{
    std::vector<uchar4> pixels;
    unsigned int width = 0;
    unsigned int height = 0;
};

// Build the OptiX pipeline and acceleration structure, render one frame, and
// copy the final pixels back to host memory.
renderResult renderScene(
    const sceneData& scene,
    const std::filesystem::path& ptxPath
);
