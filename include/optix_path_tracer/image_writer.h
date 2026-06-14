#pragma once

#include <cuda_runtime.h>

#include <filesystem>
#include <vector>

// Write an RGBA pixel buffer as a binary RGB PPM image.
void writePpm(
    const std::filesystem::path& outputPath,
    const std::vector<uchar4>& pixels,
    unsigned int width,
    unsigned int height
);
