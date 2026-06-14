#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "optix_path_tracer/error_check.h"

// Owns a typed CUDA allocation and releases it when the wrapper leaves scope.
// Copying is disabled so that a device pointer always has a single owner.
template <typename valueType>
class cudaBuffer
{
public:
    cudaBuffer() = default;
    cudaBuffer(const cudaBuffer&) = delete;
    cudaBuffer& operator=(const cudaBuffer&) = delete;

    ~cudaBuffer()
    {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    void allocate(std::size_t valueCount)
    {
        // cudaMalloc accepts a byte count even though this wrapper exposes a
        // typed pointer to the rest of the renderer.
        checkCuda(
            cudaMalloc(
                reinterpret_cast<void**>(&pointer_),
                valueCount * sizeof(valueType)
            ),
            "cudaMalloc"
        );
    }

    valueType* data()
    {
        return pointer_;
    }

    const valueType* data() const
    {
        return pointer_;
    }

    CUdeviceptr devicePointer() const
    {
        // OptiX host APIs use the CUDA driver API's integer pointer type.
        return reinterpret_cast<CUdeviceptr>(pointer_);
    }

private:
    valueType* pointer_ = nullptr;
};
