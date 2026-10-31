# Riemann2D

Riemann2D is a self-contained C++17 rigid-body and multiphysics playground for
the XY plane. The implementation is independent from the 3D Riemann project;
that repository and Box2D are used only as architectural and algorithmic
references.

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
- Dependency-free unit and integration tests.

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
extinction shading. The air scene retains a 480 x 360 grid. Liquid rendering
can also switch to particles.

Liquid and smoke have CPU, CUDA, and Compute Shader solver backends. Each scene
has independent backend buttons, device status, and solver timing. The viewer
prefers CUDA when available, then native Compute Shader, then CPU. Switching
preserves the current simulation, and runtime failures fall back to CPU. The
library itself defaults to CPU.

Compute Shader uses DX11 Shader Model 5 on Windows and Metal on macOS; it does not
require the CUDA Toolkit or an NVIDIA GPU. See [algorithm references and
measurements](Docs/FluidOptimization.md), [CUDA setup and validation](Docs/CudaFluids.md)
and [Compute Shader implementation, measurements, and platform validation](Docs/ComputeFluids.md).

## Collision terminology

The broadphase uses sweep and prune (SAP). The separating axis theorem (SAT)
is a narrowphase test; GJK/EPA handles the general support-mapped convex cases,
and triangle meshes dispatch to their child triangles.
