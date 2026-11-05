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

Snow and sand use quadratic B-spline Material Point Method transfers with
PIC/FLIP, APIC, and MLS-MPM variants. Fixed-corotated snow elasticity with
compression hardening blends with Hencky elasticity and Drucker-Prager sand
plasticity. The deformation solver includes scalar CPU, SSE2/NEON, CUDA, DX11
Compute Shader, and native Metal transfer paths, CFL-limited integration,
viscous stress, and moving capsule friction. GPU grid transfers gather particles
from spatial bins without contended floating-point atomics.

## Viewer

The left immediate-mode panel selects a scene and changes gravity, contact
solver, and solver backend. The right viewport renders the XY plane through an
orthographic camera. Rendering owns all platform RHI objects on a dedicated
thread and consumes immutable, monotonically numbered frame packets.

Available scenes are Shape Gallery, Joint Chain, Rope XPBD,
Fluid Gravity, Air No Gravity, Destruction, Ragdoll, and Snow.

The snow scene couples a surface-following capsule with Coulomb friction.
A robust local free-surface estimate ahead of the tool sets its penetration to
20 percent of the current snow thickness. Resampling follows horizontal travel,
so a stationary tool does not progressively drill into its own trench. Displaced
material persists between passes. Viscosity controls rate-dependent stress. Real-time
quality uses a coarse grid and a bounded solver time budget; high-detail quality
uses a denser grid and more material samples. Density-surface rendering adds
granular texture, blue snow shadows, and contour lighting.

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
require the CUDA Toolkit or an NVIDIA GPU.

## Collision terminology

The broadphase uses sweep and prune (SAP). The separating axis theorem (SAT)
is a narrowphase test; GJK/EPA handles the general support-mapped convex cases,
and triangle meshes dispatch to their child triangles.
