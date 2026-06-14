cmake_minimum_required(VERSION 3.24)

project(OptiXGPUPathTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(
    OPTIX_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/optix-dev"
    CACHE PATH
    "Path to the NVIDIA OptiX SDK or optix-dev repository"
)
find_path(
    OPTIX_INCLUDE_DIR
    NAMES optix.h
    HINTS
        "${OPTIX_ROOT}"
        "$ENV{OPTIX_ROOT}"
        "$ENV{OptiX_INSTALL_DIR}"
    PATH_SUFFIXES include
)

if(NOT OPTIX_INCLUDE_DIR)
    message(FATAL_ERROR
        "OptiX headers were not found. Initialize third_party/optix-dev or set "
        "OPTIX_ROOT to an NVIDIA OptiX SDK directory."
    )
endif()

set(DEVICE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/device/device.cu")
set(DEVICE_PTX "${CMAKE_CURRENT_BINARY_DIR}/device.ptx")

add_custom_command(
    OUTPUT "${DEVICE_PTX}"
    COMMAND "${CMAKE_CUDA_COMPILER}"
        --ptx
        --std=c++17
        --use_fast_math
        --gpu-architecture=compute_75
        -I"${OPTIX_INCLUDE_DIR}"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
        -o "${DEVICE_PTX}"
        "${DEVICE_SOURCE}"
    DEPENDS
        "${DEVICE_SOURCE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/optix_path_tracer/shared_types.h"
    VERBATIM
)

add_custom_target(optix_device_ptx DEPENDS "${DEVICE_PTX}")

add_executable(
    optix_path_tracer
    src/image_writer.cpp
    src/main.cpp
    src/obj_loader.cpp
    src/optix_renderer.cpp
    src/optix_function_table.cpp
    src/scene_loader.cpp
)

add_dependencies(optix_path_tracer optix_device_ptx)

target_include_directories(
    optix_path_tracer
    PRIVATE
        "${OPTIX_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/single_include"
)

target_link_libraries(
    optix_path_tracer
    PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
)

add_custom_command(
    TARGET optix_path_tracer POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${DEVICE_PTX}"
        "$<TARGET_FILE_DIR:optix_path_tracer>/device.ptx"
    VERBATIM
)
