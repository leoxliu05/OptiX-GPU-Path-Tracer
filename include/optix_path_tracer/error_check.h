#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include <stdexcept>
#include <string>

inline void checkCuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result)
        );
    }
}

inline void checkOptix(OptixResult result, const char* operation)
{
    if (result != OPTIX_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": OptiX error " + std::to_string(result)
        );
    }
}

