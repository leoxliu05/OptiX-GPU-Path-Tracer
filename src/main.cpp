#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "LaunchParams.h"

namespace {

void checkCuda(cudaError_t result, const char* expression)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(expression) + ": " + cudaGetErrorString(result)
        );
    }
}

void checkOptix(OptixResult result, const char* expression)
{
    if (result != OPTIX_SUCCESS) {
        throw std::runtime_error(
            std::string(expression) + ": OptiX error " + std::to_string(result)
        );
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)
#define OPTIX_CHECK(call) checkOptix((call), #call)

void contextLog(unsigned int level, const char* tag, const char* message, void*)
{
    std::cerr << "[OptiX][" << level << "][" << tag << "] " << message;
}

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

template <typename T>
struct DeviceBuffer
{
    T* pointer = nullptr;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
        if (pointer) {
            cudaFree(pointer);
        }
    }

    void allocate(std::size_t count)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
    }

    CUdeviceptr devicePointer() const
    {
        return reinterpret_cast<CUdeviceptr>(pointer);
    }
};

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
{
    std::array<char, OPTIX_SBT_RECORD_HEADER_SIZE> header{};
    T data{};
};

struct EmptySbtData
{
};

OptixModule createModule(
    OptixDeviceContext context,
    const OptixModuleCompileOptions& moduleOptions,
    const OptixPipelineCompileOptions& pipelineOptions,
    const std::string& ptx
)
{
    OptixModule module = nullptr;
    std::array<char, 4096> log{};
    std::size_t logSize = log.size();

#if OPTIX_VERSION >= 70700
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
#else
    const OptixResult result = optixModuleCreateFromPTX(
        context,
        &moduleOptions,
        &pipelineOptions,
        ptx.data(),
        ptx.size(),
        log.data(),
        &logSize,
        &module
    );
#endif

    if (logSize > 1) {
        std::cerr << log.data() << '\n';
    }
    OPTIX_CHECK(result);
    return module;
}

OptixProgramGroup createProgramGroup(
    OptixDeviceContext context,
    const OptixProgramGroupDesc& description
)
{
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
    OPTIX_CHECK(result);
    return group;
}

void writePpm(
    const std::filesystem::path& path,
    const std::vector<uchar4>& pixels,
    unsigned int width,
    unsigned int height
)
{
    std::ofstream output(path, std::ios::binary);
    output << "P6\n" << width << ' ' << height << "\n255\n";
    for (const uchar4 pixel : pixels) {
        output.put(static_cast<char>(pixel.x));
        output.put(static_cast<char>(pixel.y));
        output.put(static_cast<char>(pixel.z));
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const std::filesystem::path executable = std::filesystem::absolute(argv[0]);
        const std::filesystem::path ptxPath =
            executable.parent_path() / "device.ptx";
        const std::filesystem::path outputPath = argc > 1
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("optix_triangle.ppm");

        constexpr unsigned int width = 960;
        constexpr unsigned int height = 540;

        CUDA_CHECK(cudaFree(nullptr));
        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions contextOptions{};
        contextOptions.logCallbackFunction = contextLog;
#ifndef NDEBUG
        contextOptions.logCallbackLevel = 4;
        contextOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
        contextOptions.logCallbackLevel = 2;
#endif

        OptixDeviceContext context = nullptr;
        OPTIX_CHECK(optixDeviceContextCreate(nullptr, &contextOptions, &context));

        const std::string ptx = readTextFile(ptxPath);

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
        pipelineOptions.usesMotionBlur = false;
        pipelineOptions.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineOptions.numPayloadValues = 3;
        pipelineOptions.numAttributeValues = 2;
        pipelineOptions.exceptionFlags =
            OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
            OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
        pipelineOptions.pipelineLaunchParamsVariableName = "params";

        OptixModule module = createModule(
            context,
            moduleOptions,
            pipelineOptions,
            ptx
        );

        OptixProgramGroupDesc raygenDescription{};
        raygenDescription.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenDescription.raygen.module = module;
        raygenDescription.raygen.entryFunctionName = "__raygen__render";
        OptixProgramGroup raygenGroup = createProgramGroup(
            context,
            raygenDescription
        );

        OptixProgramGroupDesc missDescription{};
        missDescription.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDescription.miss.module = module;
        missDescription.miss.entryFunctionName = "__miss__radiance";
        OptixProgramGroup missGroup = createProgramGroup(context, missDescription);

        OptixProgramGroupDesc hitDescription{};
        hitDescription.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitDescription.hitgroup.moduleCH = module;
        hitDescription.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        OptixProgramGroup hitGroup = createProgramGroup(context, hitDescription);

        const std::array<OptixProgramGroup, 3> programGroups{
            raygenGroup,
            missGroup,
            hitGroup
        };

        OptixPipelineLinkOptions linkOptions{};
        linkOptions.maxTraceDepth = 1;

        OptixPipeline pipeline = nullptr;
        std::array<char, 4096> pipelineLog{};
        std::size_t pipelineLogSize = pipelineLog.size();
        const OptixResult pipelineResult = optixPipelineCreate(
            context,
            &pipelineOptions,
            &linkOptions,
            programGroups.data(),
            static_cast<unsigned int>(programGroups.size()),
            pipelineLog.data(),
            &pipelineLogSize,
            &pipeline
        );
        if (pipelineLogSize > 1) {
            std::cerr << pipelineLog.data() << '\n';
        }
        OPTIX_CHECK(pipelineResult);

        OptixStackSizes stackSizes{};
        for (OptixProgramGroup group : programGroups) {
#if OPTIX_VERSION >= 70700
            OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stackSizes, pipeline));
#else
            OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stackSizes));
#endif
        }

        unsigned int directCallableStackFromTraversal = 0;
        unsigned int directCallableStackFromState = 0;
        unsigned int continuationStack = 0;
        OPTIX_CHECK(optixUtilComputeStackSizes(
            &stackSizes,
            1,
            0,
            0,
            &directCallableStackFromTraversal,
            &directCallableStackFromState,
            &continuationStack
        ));
        OPTIX_CHECK(optixPipelineSetStackSize(
            pipeline,
            directCallableStackFromTraversal,
            directCallableStackFromState,
            continuationStack,
            1
        ));

        const std::array<float3, 3> vertices{
            make_float3(-0.85f, -0.65f, 0.0f),
            make_float3(0.85f, -0.65f, 0.0f),
            make_float3(0.0f, 0.85f, 0.0f)
        };
        DeviceBuffer<float3> deviceVertices;
        deviceVertices.allocate(vertices.size());
        CUDA_CHECK(cudaMemcpy(
            deviceVertices.pointer,
            vertices.data(),
            sizeof(vertices),
            cudaMemcpyHostToDevice
        ));

        const CUdeviceptr vertexPointer = deviceVertices.devicePointer();
        const unsigned int geometryFlags = OPTIX_GEOMETRY_FLAG_NONE;
        OptixBuildInput buildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
        buildInput.triangleArray.numVertices =
            static_cast<unsigned int>(vertices.size());
        buildInput.triangleArray.vertexBuffers = &vertexPointer;
        buildInput.triangleArray.flags = &geometryFlags;
        buildInput.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions accelOptions{};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes accelSizes{};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &accelOptions,
            &buildInput,
            1,
            &accelSizes
        ));

        DeviceBuffer<std::uint8_t> accelScratch;
        DeviceBuffer<std::uint8_t> accelOutput;
        accelScratch.allocate(accelSizes.tempSizeInBytes);
        accelOutput.allocate(accelSizes.outputSizeInBytes);

        OptixTraversableHandle traversable = 0;
        OPTIX_CHECK(optixAccelBuild(
            context,
            nullptr,
            &accelOptions,
            &buildInput,
            1,
            accelScratch.devicePointer(),
            accelSizes.tempSizeInBytes,
            accelOutput.devicePointer(),
            accelSizes.outputSizeInBytes,
            &traversable,
            nullptr,
            0
        ));

        SbtRecord<EmptySbtData> raygenRecord{};
        SbtRecord<EmptySbtData> missRecord{};
        SbtRecord<EmptySbtData> hitRecord{};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenGroup, &raygenRecord));
        OPTIX_CHECK(optixSbtRecordPackHeader(missGroup, &missRecord));
        OPTIX_CHECK(optixSbtRecordPackHeader(hitGroup, &hitRecord));

        DeviceBuffer<SbtRecord<EmptySbtData>> deviceRaygenRecord;
        DeviceBuffer<SbtRecord<EmptySbtData>> deviceMissRecord;
        DeviceBuffer<SbtRecord<EmptySbtData>> deviceHitRecord;
        deviceRaygenRecord.allocate(1);
        deviceMissRecord.allocate(1);
        deviceHitRecord.allocate(1);
        CUDA_CHECK(cudaMemcpy(
            deviceRaygenRecord.pointer,
            &raygenRecord,
            sizeof(raygenRecord),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            deviceMissRecord.pointer,
            &missRecord,
            sizeof(missRecord),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            deviceHitRecord.pointer,
            &hitRecord,
            sizeof(hitRecord),
            cudaMemcpyHostToDevice
        ));

        OptixShaderBindingTable sbt{};
        sbt.raygenRecord = deviceRaygenRecord.devicePointer();
        sbt.missRecordBase = deviceMissRecord.devicePointer();
        sbt.missRecordStrideInBytes = sizeof(missRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = deviceHitRecord.devicePointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(hitRecord);
        sbt.hitgroupRecordCount = 1;

        DeviceBuffer<uchar4> devicePixels;
        devicePixels.allocate(static_cast<std::size_t>(width) * height);

        LaunchParams launchParams{
            devicePixels.pointer,
            width,
            height,
            traversable
        };
        DeviceBuffer<LaunchParams> deviceLaunchParams;
        deviceLaunchParams.allocate(1);
        CUDA_CHECK(cudaMemcpy(
            deviceLaunchParams.pointer,
            &launchParams,
            sizeof(launchParams),
            cudaMemcpyHostToDevice
        ));

        OPTIX_CHECK(optixLaunch(
            pipeline,
            nullptr,
            deviceLaunchParams.devicePointer(),
            sizeof(launchParams),
            &sbt,
            width,
            height,
            1
        ));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uchar4> pixels(static_cast<std::size_t>(width) * height);
        CUDA_CHECK(cudaMemcpy(
            pixels.data(),
            devicePixels.pointer,
            pixels.size() * sizeof(uchar4),
            cudaMemcpyDeviceToHost
        ));
        writePpm(outputPath, pixels, width, height);

        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        OPTIX_CHECK(optixProgramGroupDestroy(hitGroup));
        OPTIX_CHECK(optixProgramGroupDestroy(missGroup));
        OPTIX_CHECK(optixProgramGroupDestroy(raygenGroup));
        OPTIX_CHECK(optixModuleDestroy(module));
        OPTIX_CHECK(optixDeviceContextDestroy(context));

        std::cout << "Rendered " << outputPath << " (" << width << 'x'
                  << height << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
