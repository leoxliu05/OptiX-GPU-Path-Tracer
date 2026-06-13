# OptiX GPU Path Tracer

This repository starts with a minimal, real NVIDIA OptiX renderer. It builds a
triangle acceleration structure, launches an OptiX pipeline, and writes a
960x540 PPM image with barycentric vertex colors.

## Requirements

- An NVIDIA RTX GPU and a current NVIDIA display driver
- Visual Studio with the Desktop development with C++ workload
- CMake 3.24 or newer
- NVIDIA CUDA Toolkit
- NVIDIA OptiX 9.1 headers (already vendored under `third_party/optix-dev`)

## Configure and build

On the configured Windows machine, the shortest path is:

```powershell
.\build_and_run.bat
```

That script finds Visual Studio, builds the project, and renders
`optix_triangle.ppm`.

For a manual build, open a Developer PowerShell for Visual Studio, then run:

```powershell
cmake -S . -B build -G Ninja
cmake --build build
```

The project uses NVIDIA's official `optix-dev` 9.1 headers. OptiX runtime
support is provided by the NVIDIA display driver.

## Render

```powershell
.\build\optix_triangle.exe .\optix_triangle.ppm
```

The result is written as a binary PPM image. Most image editors can open PPM,
and ImageMagick can convert it to PNG:

```powershell
magick .\optix_triangle.ppm .\optix_triangle.png
```

## Next milestones

1. Add a camera and progressive accumulation.
2. Add material data to hit-group SBT records.
3. Implement diffuse path bounces and Russian roulette.
4. Add direct-light sampling and multiple importance sampling.
5. Load glTF scenes and integrate the OptiX denoiser.
