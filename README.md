# Introduction

Riemann2D is a 2D rigid-body and multiphysics engine with zero dependencies. It is the 2D version of [Riemann](https://github.com/atlantis13579/Riemann).

## Features

* Simulation and Collision Detection for various shapes
    * Box, circle, capsule, convex polygon, and triangle mesh
    * Static BVH, dynamic fat-AABB tree, and sweep and prune
    * SAT, GJK, and EPA narrowphase

* Rigid Body Dynamics Simulation
    * Sequential impulse and active-set LCP contact solvers
    * Distance, revolute, prismatic, weld, wheel, motor, mouse, pulley, gear, and friction joints
    * Articulated ragdolls
    * Scalar CPU and SSE2/NEON linear algebra backends

* Destruction Simulation
    * Recursive polygon fracture

* Deformation Body Simulation
    * XPBD rope simulation

* Fluid Dynamics Simulation
    * Spatially hashed position-based liquid
    * SPH density kernels with position-based fluid (PBF) constraints
    * Pressure field interaction with rigid bodies

* Aerodynamics Simulation
    * MAC-grid smoke with limited MacCormack transport and multigrid pressure projection
    * Surface pressure and incident-flow drag on rigid bodies
    * Moving solid boundaries with masked pressure correction, matched normal velocity, and smoke displacement

* Continuum Media Simulation
    * Quadratic B-spline MPM with PIC/FLIP, APIC, and MLS-MPM transfers
    * Fixed-corotated snow elasticity, compression hardening, Hencky elasticity, and Drucker-Prager sand plasticity
    * CFL-limited integration, viscous stress, and moving capsule friction

## Examples

#### 2D Rigid Body Dynamics

https://github.com/user-attachments/assets/cc19f7b5-238d-4bad-9e03-2220b26acb19

#### 2D Articulated Dynamics

https://github.com/user-attachments/assets/dd925bca-e1ca-4a34-906c-ba2994c3c1a7

#### 2D Destruction System

https://github.com/user-attachments/assets/770ba12a-4f38-490c-a4ce-50841d9cab56

#### 2D Fluids Simulation

https://github.com/user-attachments/assets/4e08dce5-ad5f-4f5e-91f4-c00a8e93818b

#### 2D Aerodynamics Simulation

https://github.com/user-attachments/assets/eb9d1fac-28bb-47ef-a8b1-536ad06330ee

#### 2D Snow Simulation

https://github.com/user-attachments/assets/2b9598ca-af02-47cf-887c-d2fcb14593c4
