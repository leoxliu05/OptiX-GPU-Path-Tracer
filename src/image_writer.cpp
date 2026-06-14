#include "optix_path_tracer/image_writer.h"

#include <fstream>
#include <stdexcept>

void writePpm(
    const std::filesystem::path& outputPath,
    const std::vector<uchar4>& pixels,
    unsigned int width,
    unsigned int height
)
{
    std::ofstream output(outputPath, std::ios::binary);
    if (!output) {
        throw std::runtime_error(
            "Could not open output image: " + outputPath.string()
        );
    }

    output << "P6\n" << width << ' ' << height << "\n255\n";
    for (const uchar4 pixel : pixels) {
        output.put(static_cast<char>(pixel.x));
        output.put(static_cast<char>(pixel.y));
        output.put(static_cast<char>(pixel.z));
    }
}
