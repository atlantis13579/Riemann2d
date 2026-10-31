# Riemann2D

Riemann2D is a self-contained C++17 rigid-body and multiphysics playground for
the XY plane. The implementation is independent from the 3D Riemann project;
that repository and Box2D are used only as architectural and algorithmic
references.

## Layout

- `Src` - engine code and public headers.
- `Viewer` - scene browser, render thread, and platform RHI backends.
- `Contents` - scene descriptions and runtime assets.
- `Test` - dependency-free unit and integration tests.

## Build

```text
cmake -S Build -B Build/Win64 -G "Visual Studio 17 2022" -A x64
cmake --build Build/Win64 --config Debug
ctest --test-dir Build/Win64 -C Debug --output-on-failure
```

The Windows viewer uses Direct3D 11. The macOS viewer uses Metal. Tests and
the physics library build on any C++17 platform without a graphics SDK.

## Implemented systems

- Five collision shapes: Box, Sphere, Capsule, Convex, and TriMesh, including
  every unordered pair combination.
- Static BVH, generation-safe dynamic fat-AABB tree, SAP broadphase, and
  SAT/GJK/EPA narrowphase dispatch.
- Rigid bodies with Sequential Impulse or dense active-set LCP contact solving.
- Scalar CPU and SSE2/NEON SIMD linear algebra backends.
- Distance, Revolute, Prismatic, Weld, Wheel, Motor, Mouse, Pulley, Gear, and
  Friction constraints.
- XPBD rope, spatially hashed position-based liquid, gravity-free MAC air field,
  recursive polygon fracture, and articulated ragdolls.
- Dependency-free tests with an optional name filter:

```text
Binaries/Win64/Debug/Riemann2DTests.exe Collision
```

## Viewer

The left immediate-mode panel selects a scene and changes gravity, contact
solver, and solver backend. The right viewport renders the XY plane through an
orthographic camera. Rendering owns all platform RHI objects on a dedicated
thread and consumes immutable, monotonically numbered frame packets.

Available scenes are Shape Gallery, Joint Chain, Rope XPBD,
Fluid Gravity, Air No Gravity, Destruction, and Ragdoll.

Liquid surfaces use a filtered density texture with antialiased contours,
depth tint, and surface highlights, including isolated droplets. Smoke uses
limited MacCormack transport, multigrid pressure projection, and filtered
extinction shading. The air scene retains a 480 x 360 grid; click the viewport
to inject smoke, or toggle velocity vectors for diagnostics. Liquid rendering
can still switch to particles, and tank rotation/body dragging remain available.

Release performance and DX11 framebuffer verification tools:

```text
Binaries/Win64/Release/FluidBenchmark.exe
Binaries/Win64/Release/FluidPreview.exe Build/FluidPreview
```

The preview tool creates a hidden window and writes six PPM frames without
desktop automation. See [algorithm references and measurements](Docs/FluidOptimization.md).

Liquid and smoke have CPU, CUDA, and Compute Shader solver backends. Each scene
has independent backend buttons, device status, and solver timing. The viewer
prefers CUDA when available, then native Compute Shader, then CPU. Switching
preserves the current simulation, and runtime failures fall back to CPU. The
library itself defaults to CPU; select `FluidComputeBackend::CUDA` or
`FluidComputeBackend::ComputeShader` with `SetBackend` to opt into GPU execution.

CMake detects the CUDA Toolkit automatically. Use `-DRIEMANN2D_ENABLE_CUDA=OFF`
for a build without CUDA dependencies. See [CUDA setup and validation](Docs/CudaFluids.md).

Compute Shader uses DX11 Shader Model 5 on Windows and Metal on macOS; it does not
require the CUDA Toolkit or an NVIDIA GPU. Disable it with
`-DRIEMANN2D_ENABLE_COMPUTE_SHADER=OFF`. See [Compute Shader implementation,
measurements, and platform validation](Docs/ComputeFluids.md).

```text
Binaries/Win64/Release/FluidBenchmark.exe --cuda
Binaries/Win64/Release/FluidPreview.exe Build/CudaPreview --cuda
Binaries/Win64/Release/FluidBenchmark.exe --compute
Binaries/Win64/Release/FluidPreview.exe Build/ComputePreview --compute
Binaries/Win64/Release/ViewerBackendPreview.exe
```

The Metal/Cocoa sources implement the corresponding rendering and compute paths.
DX11 was built and tested on Windows; Metal compilation and runtime validation
still require a macOS machine.

## Collision terminology

The broadphase uses sweep and prune (SAP). The separating axis theorem (SAT)
is a narrowphase test; GJK/EPA handles the general support-mapped convex cases,
and triangle meshes dispatch to their child triangles.
