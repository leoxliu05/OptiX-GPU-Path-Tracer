#pragma once

#include <filesystem>

#include "optix_path_tracer/scene.h"

// Parse a scene description and load all referenced mesh data.
sceneData loadScene(const std::filesystem::path& scenePath);
