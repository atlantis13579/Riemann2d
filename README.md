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
- XPBD rope, gravity-driven particle fluid, gravity-free Eulerian air field,
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

The Metal/Cocoa sources are kept at feature parity with DX11. Windows CI can
compile and smoke-test DX11; the Metal target must be compiled on macOS.

## Collision terminology

The broadphase uses sweep and prune (SAP). The separating axis theorem (SAT)
is a narrowphase test; GJK/EPA handles the general support-mapped convex cases,
and triangle meshes dispatch to their child triangles.
