#pragma once

#include <cuda_runtime.h>

#include <filesystem>
#include <vector>

#include "optix_path_tracer/scene.h"

struct renderResult
{
    std::vector<uchar4> pixels;
    unsigned int width = 0;
    unsigned int height = 0;
};

renderResult renderScene(
    const sceneData& scene,
    const std::filesystem::path& ptxPath
);
