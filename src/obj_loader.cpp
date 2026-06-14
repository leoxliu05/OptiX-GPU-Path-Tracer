#include "optix_path_tracer/obj_loader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

// Apply the limited scale-and-translation transform while loading vertices so
// the renderer only needs one world-space vertex buffer.
float3 applyTransform(float3 vertex, const objectTransform& transform)
{
    return make_float3(
        vertex.x * transform.scale.x + transform.translation.x,
        vertex.y * transform.scale.y + transform.translation.y,
        vertex.z * transform.scale.z + transform.translation.z
    );
}

// OBJ indices are one-based and may be negative. Texture-coordinate and normal
// suffixes are ignored because the current renderer only needs positions.
std::size_t parseVertexIndex(const std::string& token, std::size_t vertexCount)
{
    const std::size_t slashPosition = token.find('/');
    const int index = std::stoi(token.substr(0, slashPosition));
    const int resolvedIndex = index > 0
        ? index - 1
        : static_cast<int>(vertexCount) + index;
    if (resolvedIndex < 0 || resolvedIndex >= static_cast<int>(vertexCount)) {
        throw std::runtime_error("OBJ face contains an invalid vertex index");
    }
    return static_cast<std::size_t>(resolvedIndex);
}

} // namespace

std::vector<float3> loadObjTriangles(
    const std::filesystem::path& path,
    const objectTransform& transform
)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open OBJ file: " + path.string());
    }

    std::vector<float3> positions;
    std::vector<float3> triangles;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream lineStream(line);
        std::string recordType;
        lineStream >> recordType;

        if (recordType == "v") {
            // Transform each position once instead of repeating the operation
            // for every ray intersection on the GPU.
            float3 vertex{};
            lineStream >> vertex.x >> vertex.y >> vertex.z;
            positions.push_back(applyTransform(vertex, transform));
            continue;
        }

        if (recordType != "f") {
            continue;
        }

        std::vector<std::size_t> faceIndices;
        std::string token;
        while (lineStream >> token) {
            faceIndices.push_back(parseVertexIndex(token, positions.size()));
        }
        if (faceIndices.size() < 3) {
            throw std::runtime_error("OBJ face has fewer than three vertices: " + path.string());
        }

        // Convert an arbitrary polygon into a triangle fan rooted at vertex 0.
        for (std::size_t index = 1; index + 1 < faceIndices.size(); ++index) {
            triangles.push_back(positions[faceIndices[0]]);
            triangles.push_back(positions[faceIndices[index]]);
            triangles.push_back(positions[faceIndices[index + 1]]);
        }
    }

    return triangles;
}
