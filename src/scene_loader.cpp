#include "optix_path_tracer/scene_loader.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "optix_path_tracer/obj_loader.h"

namespace {

using json = nlohmann::json;

float3 readFloat3(const json& value, const std::string& fieldName)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(fieldName + " must contain three numbers");
    }
    return make_float3(
        value.at(0).get<float>(),
        value.at(1).get<float>(),
        value.at(2).get<float>()
    );
}

objectTransform readTransform(const json& value)
{
    objectTransform transform;
    if (value.contains("scale")) {
        transform.scale = readFloat3(value.at("scale"), "transform.scale");
    }
    if (value.contains("translation")) {
        transform.translation = readFloat3(
            value.at("translation"),
            "transform.translation"
        );
    }
    return transform;
}

objectTransform combineTransforms(
    const objectTransform& parent,
    const objectTransform& child
)
{
    objectTransform combined;
    combined.scale = make_float3(
        parent.scale.x * child.scale.x,
        parent.scale.y * child.scale.y,
        parent.scale.z * child.scale.z
    );
    combined.translation = make_float3(
        parent.translation.x + child.translation.x * parent.scale.x,
        parent.translation.y + child.translation.y * parent.scale.y,
        parent.translation.z + child.translation.z * parent.scale.z
    );
    return combined;
}

} // namespace

sceneData loadScene(const std::filesystem::path& scenePath)
{
    std::ifstream input(scenePath);
    if (!input) {
        throw std::runtime_error("Could not open scene file: " + scenePath.string());
    }

    json document;
    input >> document;
    sceneData scene;

    const json& renderer = document.at("renderer");
    scene.renderer.width = renderer.at("width").get<unsigned int>();
    scene.renderer.height = renderer.at("height").get<unsigned int>();
    scene.renderer.samplesPerPixel =
        renderer.at("samplesPerPixel").get<unsigned int>();
    scene.renderer.maxDepth = renderer.at("maxDepth").get<unsigned int>();
    scene.renderer.seed = renderer.value("seed", 1u);
    scene.renderer.outputPath = renderer.value("output", "render.ppm");

    const json& camera = document.at("camera");
    scene.camera.position = readFloat3(camera.at("position"), "camera.position");
    scene.camera.target = readFloat3(camera.at("target"), "camera.target");
    scene.camera.up = readFloat3(camera.at("up"), "camera.up");
    scene.camera.verticalFovDegrees = camera.at("verticalFovDegrees").get<float>();
    scene.environmentColor = readFloat3(
        document.value("environmentColor", json::array({0.0f, 0.0f, 0.0f})),
        "environmentColor"
    );

    std::unordered_map<std::string, std::uint32_t> materialLookup;
    for (const json& material : document.at("materials")) {
        const std::string name = material.at("name").get<std::string>();
        if (materialLookup.count(name) != 0) {
            throw std::runtime_error("Duplicate material name: " + name);
        }
        materialLookup[name] = static_cast<std::uint32_t>(scene.materials.size());
        scene.materials.push_back(materialData{
            readFloat3(material.at("albedo"), "materials.albedo"),
            readFloat3(
                material.value("emission", json::array({0.0f, 0.0f, 0.0f})),
                "materials.emission"
            )
        });
    }

    const objectTransform rootTransform = document.contains("rootTransform")
        ? readTransform(document.at("rootTransform"))
        : objectTransform{};
    const std::filesystem::path sceneDirectory = scenePath.parent_path();

    for (const json& mesh : document.at("meshes")) {
        const std::string materialName = mesh.at("material").get<std::string>();
        const auto materialIterator = materialLookup.find(materialName);
        if (materialIterator == materialLookup.end()) {
            throw std::runtime_error("Unknown material: " + materialName);
        }

        const objectTransform meshTransform = mesh.contains("transform")
            ? readTransform(mesh.at("transform"))
            : objectTransform{};
        const std::vector<float3> meshVertices = loadObjTriangles(
            sceneDirectory / mesh.at("path").get<std::string>(),
            combineTransforms(rootTransform, meshTransform)
        );
        scene.vertices.insert(
            scene.vertices.end(),
            meshVertices.begin(),
            meshVertices.end()
        );
        scene.materialIndices.insert(
            scene.materialIndices.end(),
            meshVertices.size() / 3,
            materialIterator->second
        );
    }

    if (scene.vertices.empty()) {
        throw std::runtime_error("The scene does not contain any triangles");
    }
    return scene;
}
