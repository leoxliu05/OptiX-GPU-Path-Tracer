#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "optix_path_tracer/error_check.h"

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
        return reinterpret_cast<CUdeviceptr>(pointer_);
    }

private:
    valueType* pointer_ = nullptr;
};
