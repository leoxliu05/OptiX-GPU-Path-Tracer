#include "optix_path_tracer/optix_renderer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "optix_path_tracer/cuda_buffer.h"
#include "optix_path_tracer/error_check.h"
#include "optix_path_tracer/math_utils.h"
#include "optix_path_tracer/shared_types.h"

namespace {

// Every SBT record starts with an OptiX-generated header followed by data that
// becomes available through optixGetSbtDataPointer() on the device.
template <typename dataType>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) sbtRecord
{
    std::array<char, OPTIX_SBT_RECORD_HEADER_SIZE> header{};
    dataType data{};
};

struct emptySbtData
{
};

// Own the OptiX objects created for one render. Destruction happens in reverse
// dependency order when renderScene returns or throws an exception.
struct optixResources
{
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup raygenGroup = nullptr;
    OptixProgramGroup missGroup = nullptr;
    OptixProgramGroup hitGroup = nullptr;
    OptixPipeline pipeline = nullptr;

    ~optixResources()
    {
        if (pipeline != nullptr) {
            optixPipelineDestroy(pipeline);
        }
        if (hitGroup != nullptr) {
            optixProgramGroupDestroy(hitGroup);
        }
        if (missGroup != nullptr) {
            optixProgramGroupDestroy(missGroup);
        }
        if (raygenGroup != nullptr) {
            optixProgramGroupDestroy(raygenGroup);
        }
        if (module != nullptr) {
            optixModuleDestroy(module);
        }
        if (context != nullptr) {
            optixDeviceContextDestroy(context);
        }
    }
};

void logOptixMessage(
    unsigned int level,
    const char* tag,
    const char* message,
    void*
)
{
    std::cerr << "[OptiX][" << level << "][" << tag << "] " << message;
}

// PTX is text consumed by optixModuleCreate at runtime.
std::string readTextFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open PTX file: " + path.string());
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

OptixModule createModule(
    OptixDeviceContext context,
    const OptixModuleCompileOptions& moduleOptions,
    const OptixPipelineCompileOptions& pipelineOptions,
    const std::string& ptx
)
{
    // Preserve the compiler log even on success because it can contain useful
    // stack or optimization diagnostics.
    OptixModule module = nullptr;
    std::array<char, 4096> log{};
    std::size_t logSize = log.size();
    const OptixResult result = optixModuleCreate(
        context,
        &moduleOptions,
        &pipelineOptions,
        ptx.data(),
        ptx.size(),
        log.data(),
        &logSize,
        &module
    );
    if (logSize > 1) {
        std::cerr << log.data() << '\n';
    }
    checkOptix(result, "optixModuleCreate");
    return module;
}

OptixProgramGroup createProgramGroup(
    OptixDeviceContext context,
    const OptixProgramGroupDesc& description
)
{
    // A program group associates one OptiX stage with an entry point from PTX.
    OptixProgramGroupOptions options{};
    OptixProgramGroup group = nullptr;
    std::array<char, 4096> log{};
    std::size_t logSize = log.size();
    const OptixResult result = optixProgramGroupCreate(
        context,
        &description,
        1,
        &options,
        log.data(),
        &logSize,
        &group
    );
    if (logSize > 1) {
        std::cerr << log.data() << '\n';
    }
    checkOptix(result, "optixProgramGroupCreate");
    return group;
}

void createPipeline(optixResources& resources, const std::string& ptx)
{
    OptixModuleCompileOptions moduleOptions{};
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif

    OptixPipelineCompileOptions pipelineOptions{};

    // The current scene builds one geometry acceleration structure and uses two
    // payload registers to carry a 64-bit pathPayload pointer.
    pipelineOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineOptions.numPayloadValues = 2;
    pipelineOptions.numAttributeValues = 2;
    pipelineOptions.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
        OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";

    resources.module = createModule(
        resources.context,
        moduleOptions,
        pipelineOptions,
        ptx
    );

    // Register the three device entry points used by the single radiance ray type.
    OptixProgramGroupDesc raygenDescription{};
    raygenDescription.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDescription.raygen.module = resources.module;
    raygenDescription.raygen.entryFunctionName = "__raygen__render";
    resources.raygenGroup = createProgramGroup(
        resources.context,
        raygenDescription
    );

    OptixProgramGroupDesc missDescription{};
    missDescription.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDescription.miss.module = resources.module;
    missDescription.miss.entryFunctionName = "__miss__radiance";
    resources.missGroup = createProgramGroup(
        resources.context,
        missDescription
    );

    OptixProgramGroupDesc hitDescription{};
    hitDescription.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDescription.hitgroup.moduleCH = resources.module;
    hitDescription.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    resources.hitGroup = createProgramGroup(
        resources.context,
        hitDescription
    );

    const std::array<OptixProgramGroup, 3> groups{
        resources.raygenGroup,
        resources.missGroup,
        resources.hitGroup
    };

    // Paths are iterated inside raygen, so each optixTrace call only needs one
    // level of OptiX continuation depth.
    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = 1;
    std::array<char, 4096> log{};
    std::size_t logSize = log.size();
    const OptixResult pipelineResult = optixPipelineCreate(
        resources.context,
        &pipelineOptions,
        &linkOptions,
        groups.data(),
        static_cast<unsigned int>(groups.size()),
        log.data(),
        &logSize,
        &resources.pipeline
    );
    if (logSize > 1) {
        std::cerr << log.data() << '\n';
    }
    checkOptix(pipelineResult, "optixPipelineCreate");

    // OptiX requires explicit stack sizing after all program groups are linked.
    OptixStackSizes stackSizes{};
    for (OptixProgramGroup group : groups) {
        checkOptix(
            optixUtilAccumulateStackSizes(group, &stackSizes, resources.pipeline),
            "optixUtilAccumulateStackSizes"
        );
    }
    unsigned int traversalStackSize = 0;
    unsigned int stateStackSize = 0;
    unsigned int continuationStackSize = 0;
    checkOptix(
        optixUtilComputeStackSizes(
            &stackSizes,
            1,
            0,
            0,
            &traversalStackSize,
            &stateStackSize,
            &continuationStackSize
        ),
        "optixUtilComputeStackSizes"
    );
    checkOptix(
        optixPipelineSetStackSize(
            resources.pipeline,
            traversalStackSize,
            stateStackSize,
            continuationStackSize,
            1
        ),
        "optixPipelineSetStackSize"
    );
}

launchParams createLaunchParams(
    const sceneData& scene,
    uchar4* image,
    OptixTraversableHandle traversable
)
{
    // Convert the camera look-at description into a scaled orthonormal basis.
    // Device code combines W + screenX * U + screenY * V for each primary ray.
    const float3 cameraForward = normalize(subtract(
        scene.camera.target,
        scene.camera.position
    ));
    const float3 cameraRight = normalize(cross(cameraForward, scene.camera.up));
    const float3 cameraUp = normalize(cross(cameraRight, cameraForward));
    const float verticalScale = std::tan(
        scene.camera.verticalFovDegrees *
        0.5f *
        3.14159265358979323846f /
        180.0f
    );
    const float aspectRatio =
        static_cast<float>(scene.renderer.width) / scene.renderer.height;

    launchParams parameters{};
    parameters.image = image;
    parameters.width = scene.renderer.width;
    parameters.height = scene.renderer.height;
    parameters.samplesPerPixel = scene.renderer.samplesPerPixel;
    parameters.maxDepth = scene.renderer.maxDepth;
    parameters.seed = scene.renderer.seed;
    parameters.cameraOrigin = scene.camera.position;
    parameters.cameraU = multiply(
        cameraRight,
        verticalScale * aspectRatio
    );
    parameters.cameraV = multiply(cameraUp, verticalScale);
    parameters.cameraW = cameraForward;
    parameters.environmentColor = scene.environmentColor;
    parameters.traversable = traversable;
    return parameters;
}

} // namespace

renderResult renderScene(
    const sceneData& scene,
    const std::filesystem::path& ptxPath
)
{
    // cudaFree(nullptr) initializes the CUDA runtime without allocating memory.
    checkCuda(cudaFree(nullptr), "initialize CUDA runtime");
    checkOptix(optixInit(), "optixInit");

    // The context targets the current CUDA device. Debug builds enable OptiX
    // validation to report invalid pipeline and acceleration-structure usage.
    optixResources resources;
    OptixDeviceContextOptions contextOptions{};
    contextOptions.logCallbackFunction = logOptixMessage;
#ifndef NDEBUG
    contextOptions.logCallbackLevel = 4;
    contextOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
    contextOptions.logCallbackLevel = 1;
#endif
    checkOptix(
        optixDeviceContextCreate(nullptr, &contextOptions, &resources.context),
        "optixDeviceContextCreate"
    );
    createPipeline(resources, readTextFile(ptxPath));

    // Upload the flat world-space triangle array produced by the OBJ loader.
    cudaBuffer<float3> deviceVertices;
    deviceVertices.allocate(scene.vertices.size());
    checkCuda(
        cudaMemcpy(
            deviceVertices.data(),
            scene.vertices.data(),
            scene.vertices.size() * sizeof(float3),
            cudaMemcpyHostToDevice
        ),
        "copy scene vertices"
    );

    // Each triangle stores an SBT record index selecting its material.
    cudaBuffer<std::uint32_t> deviceMaterialIndices;
    deviceMaterialIndices.allocate(scene.materialIndices.size());
    checkCuda(
        cudaMemcpy(
            deviceMaterialIndices.data(),
            scene.materialIndices.data(),
            scene.materialIndices.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice
        ),
        "copy material indices"
    );

    // Describe one non-indexed triangle build input. Vertices 0..2 form the
    // first primitive, vertices 3..5 form the second, and so on.
    const CUdeviceptr vertexPointer = deviceVertices.devicePointer();
    const std::vector<unsigned int> geometryFlags(
        scene.materials.size(),
        OPTIX_GEOMETRY_FLAG_NONE
    );
    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    buildInput.triangleArray.numVertices =
        static_cast<unsigned int>(scene.vertices.size());
    buildInput.triangleArray.vertexBuffers = &vertexPointer;
    buildInput.triangleArray.flags = geometryFlags.data();
    buildInput.triangleArray.numSbtRecords =
        static_cast<unsigned int>(scene.materials.size());
    buildInput.triangleArray.sbtIndexOffsetBuffer =
        deviceMaterialIndices.devicePointer();
    buildInput.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(std::uint32_t);
    buildInput.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(std::uint32_t);

    // Query allocation sizes before building the geometry acceleration
    // structure (GAS). Scratch memory is temporary; output memory owns the BVH.
    OptixAccelBuildOptions accelerationOptions{};
    accelerationOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelerationOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes accelerationSizes{};
    checkOptix(
        optixAccelComputeMemoryUsage(
            resources.context,
            &accelerationOptions,
            &buildInput,
            1,
            &accelerationSizes
        ),
        "optixAccelComputeMemoryUsage"
    );

    cudaBuffer<std::uint8_t> accelerationScratch;
    cudaBuffer<std::uint8_t> accelerationOutput;
    accelerationScratch.allocate(accelerationSizes.tempSizeInBytes);
    accelerationOutput.allocate(accelerationSizes.outputSizeInBytes);
    OptixTraversableHandle traversable = 0;
    checkOptix(
        optixAccelBuild(
            resources.context,
            nullptr,
            &accelerationOptions,
            &buildInput,
            1,
            accelerationScratch.devicePointer(),
            accelerationSizes.tempSizeInBytes,
            accelerationOutput.devicePointer(),
            accelerationSizes.outputSizeInBytes,
            &traversable,
            nullptr,
            0
        ),
        "optixAccelBuild"
    );

    // Raygen and miss records only identify programs. Hit records additionally
    // bind the shared vertex buffer and one material to the closest-hit program.
    sbtRecord<emptySbtData> raygenRecord{};
    sbtRecord<emptySbtData> missRecord{};
    checkOptix(
        optixSbtRecordPackHeader(resources.raygenGroup, &raygenRecord),
        "pack raygen SBT record"
    );
    checkOptix(
        optixSbtRecordPackHeader(resources.missGroup, &missRecord),
        "pack miss SBT record"
    );

    std::vector<sbtRecord<hitGroupData>> hitRecords(scene.materials.size());
    for (std::size_t index = 0; index < hitRecords.size(); ++index) {
        checkOptix(
            optixSbtRecordPackHeader(resources.hitGroup, &hitRecords[index]),
            "pack hit group SBT record"
        );
        hitRecords[index].data.vertices = deviceVertices.data();
        hitRecords[index].data.material = scene.materials[index];
    }

    // SBT records must live in device memory for the duration of optixLaunch.
    cudaBuffer<sbtRecord<emptySbtData>> deviceRaygenRecord;
    cudaBuffer<sbtRecord<emptySbtData>> deviceMissRecord;
    cudaBuffer<sbtRecord<hitGroupData>> deviceHitRecords;
    deviceRaygenRecord.allocate(1);
    deviceMissRecord.allocate(1);
    deviceHitRecords.allocate(hitRecords.size());
    checkCuda(
        cudaMemcpy(
            deviceRaygenRecord.data(),
            &raygenRecord,
            sizeof(raygenRecord),
            cudaMemcpyHostToDevice
        ),
        "copy raygen SBT record"
    );
    checkCuda(
        cudaMemcpy(
            deviceMissRecord.data(),
            &missRecord,
            sizeof(missRecord),
            cudaMemcpyHostToDevice
        ),
        "copy miss SBT record"
    );
    checkCuda(
        cudaMemcpy(
            deviceHitRecords.data(),
            hitRecords.data(),
            hitRecords.size() * sizeof(sbtRecord<hitGroupData>),
            cudaMemcpyHostToDevice
        ),
        "copy hit group SBT records"
    );

    // The SBT connects the pipeline stages to the records selected by a trace.
    OptixShaderBindingTable shaderBindingTable{};
    shaderBindingTable.raygenRecord = deviceRaygenRecord.devicePointer();
    shaderBindingTable.missRecordBase = deviceMissRecord.devicePointer();
    shaderBindingTable.missRecordStrideInBytes = sizeof(missRecord);
    shaderBindingTable.missRecordCount = 1;
    shaderBindingTable.hitgroupRecordBase = deviceHitRecords.devicePointer();
    shaderBindingTable.hitgroupRecordStrideInBytes =
        sizeof(sbtRecord<hitGroupData>);
    shaderBindingTable.hitgroupRecordCount =
        static_cast<unsigned int>(hitRecords.size());

    // Allocate the output image and copy the camera, render settings, and GAS
    // handle into the launch parameter block consumed by device.cu.
    const std::size_t pixelCount =
        static_cast<std::size_t>(scene.renderer.width) * scene.renderer.height;
    cudaBuffer<uchar4> devicePixels;
    devicePixels.allocate(pixelCount);
    launchParams parameters = createLaunchParams(
        scene,
        devicePixels.data(),
        traversable
    );
    cudaBuffer<launchParams> deviceParameters;
    deviceParameters.allocate(1);
    checkCuda(
        cudaMemcpy(
            deviceParameters.data(),
            &parameters,
            sizeof(parameters),
            cudaMemcpyHostToDevice
        ),
        "copy launch parameters"
    );

    // Launch one ray-generation invocation per pixel. All samples and path
    // bounces for that pixel are evaluated inside the device program.
    checkOptix(
        optixLaunch(
            resources.pipeline,
            nullptr,
            deviceParameters.devicePointer(),
            sizeof(parameters),
            &shaderBindingTable,
            scene.renderer.width,
            scene.renderer.height,
            1
        ),
        "optixLaunch"
    );
    checkCuda(cudaDeviceSynchronize(), "wait for OptiX launch");

    // Return an owning host copy so all temporary CUDA and OptiX resources can
    // be released when this function exits.
    renderResult result;
    result.width = scene.renderer.width;
    result.height = scene.renderer.height;
    result.pixels.resize(pixelCount);
    checkCuda(
        cudaMemcpy(
            result.pixels.data(),
            devicePixels.data(),
            pixelCount * sizeof(uchar4),
            cudaMemcpyDeviceToHost
        ),
        "copy rendered pixels"
    );
    return result;
}
