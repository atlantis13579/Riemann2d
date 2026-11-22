# Introduction

Riemann2D is a 2D physics engine with zero dependencies. It is the 2D version of [Riemann](https://github.com/atlantis13579/Riemann).

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

https://github.com/user-attachments/assets/fe19c59c-058a-4796-9857-c108ccd24e3a

#### 2D Articulated Dynamics

https://github.com/user-attachments/assets/3039b23a-e292-4a3c-964f-62823611fb11

#### 2D Destruction System

https://github.com/user-attachments/assets/1d3b0390-ff7e-4d12-b607-0a9a24d82750

#### 2D Fluids Simulation

https://github.com/user-attachments/assets/e21a3cdd-ece4-491e-bf13-d0dd712b05ef

#### 2D Aerodynamics Simulation

https://github.com/user-attachments/assets/a38d7009-2112-4747-b0a1-2713b3c4ea50

#### 2D Snow Simulation

https://github.com/user-attachments/assets/d7917bd6-2ade-411c-b8be-910d29cee133


