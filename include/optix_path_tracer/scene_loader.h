#pragma once

#include <filesystem>

#include "optix_path_tracer/scene.h"

sceneData loadScene(const std::filesystem::path& scenePath);
