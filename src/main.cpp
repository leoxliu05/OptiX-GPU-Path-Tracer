#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "optix_path_tracer/image_writer.h"
#include "optix_path_tracer/optix_renderer.h"
#include "optix_path_tracer/scene_loader.h"

int main(int argumentCount, char** arguments)
{
    try {
        // Keep the executable useful without arguments while allowing any
        // scene JSON file to be selected from the command line.
        const std::filesystem::path scenePath = argumentCount > 1
            ? std::filesystem::path(arguments[1])
            : std::filesystem::path("model/cornellbox/scene.json");
        sceneData scene = loadScene(scenePath);
        if (argumentCount > 2) {
            // An explicit command-line output path takes precedence over JSON.
            scene.renderer.outputPath = arguments[2];
        }

        // CMake copies device.ptx beside the executable after a successful
        // build, so locating it does not depend on the current directory.
        const std::filesystem::path executablePath =
            std::filesystem::absolute(arguments[0]);
        const std::filesystem::path ptxPath =
            executablePath.parent_path() / "device.ptx";
        const renderResult result = renderScene(scene, ptxPath);

        // Image encoding remains separate from rendering so additional output
        // formats can be added without changing the OptiX code.
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
