#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "optix_path_tracer/image_writer.h"
#include "optix_path_tracer/optix_renderer.h"
#include "optix_path_tracer/scene_loader.h"

int main(int argumentCount, char** arguments)
{
    try {
        const std::filesystem::path scenePath = argumentCount > 1
            ? std::filesystem::path(arguments[1])
            : std::filesystem::path("model/cornellbox/scene.json");
        sceneData scene = loadScene(scenePath);
        if (argumentCount > 2) {
            scene.renderer.outputPath = arguments[2];
        }

        const std::filesystem::path executablePath =
            std::filesystem::absolute(arguments[0]);
        const std::filesystem::path ptxPath =
            executablePath.parent_path() / "device.ptx";
        const renderResult result = renderScene(scene, ptxPath);
        writePpm(
            scene.renderer.outputPath,
            result.pixels,
            result.width,
            result.height
        );

        std::cout << "Rendered " << scene.renderer.outputPath << " ("
                  << result.width << 'x' << result.height << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
