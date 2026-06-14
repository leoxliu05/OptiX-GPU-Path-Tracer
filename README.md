# OptiX GPU Path Tracer

A small, scene-driven NVIDIA OptiX path tracer. Rendering code is independent
of the bundled Cornell Box: scene assets, materials, camera settings, image
dimensions, sampling, and output paths are configured in JSON.

## Current Renderer

- Iterative diffuse path tracing on the GPU
- Cosine-weighted Lambertian scattering
- Emissive triangle meshes
- Russian roulette path termination
- OBJ mesh loading
- JSON scene descriptions
- Per-material OptiX SBT records
- ACES tone mapping and PPM output

Cook-Torrance, GGX, explicit light sampling, and MIS are intentionally not part
of this baseline.

## Build and Run

```powershell
.\build_and_run.bat
```

The script builds the renderer and runs:

```powershell
.\build\optix_path_tracer.exe .\model\cornellbox\scene.json
```

An optional second argument overrides the output path:

```powershell
.\build\optix_path_tracer.exe scene.json output.ppm
```

## Scene Format

See `model/cornellbox/scene.json` for a complete example. A scene contains:

- `renderer`: resolution, samples per pixel, maximum depth, seed, and output
- `camera`: position, target, up vector, and vertical field of view
- `environmentColor`: radiance returned by miss rays
- `rootTransform`: scale and translation applied to all meshes
- `materials`: named Lambertian albedo and optional emission
- `meshes`: OBJ paths, material names, and optional local transforms

Mesh paths are resolved relative to the JSON file, so a scene and its assets
can be moved together without changing renderer source code.

## Source Layout

- `include/optix_path_tracer/`: renderer headers and host/device shared types
- `src/device/device.cu`: OptiX device programs and diffuse path integration
- `src/scene_loader.cpp`: JSON scene loading and validation
- `src/obj_loader.cpp`: generic OBJ triangle loading
- `src/optix_renderer.cpp`: OptiX context, pipeline, GAS, SBT, and launch
- `src/image_writer.cpp`: image serialization
- `src/main.cpp`: command-line entry point
