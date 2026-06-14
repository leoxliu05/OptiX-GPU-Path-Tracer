#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include <stdexcept>
#include <string>

// Convert CUDA error codes into exceptions with operation-specific context.
inline void checkCuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result)
        );
    }
}

// Keep OptiX error handling consistent with the CUDA error path above.
inline void checkOptix(OptixResult result, const char* operation)
{
    if (result != OPTIX_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": OptiX error " + std::to_string(result)
        );
    }
}
