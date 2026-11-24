#pragma once

#include "FluidComputeBackend.h"

#include <vector>

namespace Riemann2D
{
    struct FluidParticle2D;
    struct FluidSettings2D;
    struct AerodynamicsSettings2D;
    struct Vector2;

    // Rasterized geometry and face velocities supplied by the rigid-body world.
    struct AirBoundaryData
    {
        const std::vector<int>& Solid;
        const std::vector<int>& DisplacementHead;
        const std::vector<int>& DisplacementNext;
        const std::vector<int>& BlockU;
        const std::vector<int>& BlockV;
        const std::vector<float>& VelocityU;
        const std::vector<float>& VelocityV;
    };

    class FluidGpuSolver
    {
    public:
        virtual ~FluidGpuSolver() = default;
        virtual bool Step(const FluidSettings2D& settings, float restDensity, float deltaTime,
                          int substeps, std::vector<FluidParticle2D>& particles, std::string& error) = 0;
    };

    class AirGpuSolver
    {
    public:
        virtual ~AirGpuSolver() = default;
        virtual bool Step(const AerodynamicsSettings2D& settings, const Vector2& cellSize,
                          int width, int height, float deltaTime, bool projectOnly, const AirBoundaryData* boundary, bool synchronizeOnly,
                          std::vector<float>& density, std::vector<float>& u,
                          std::vector<float>& v, std::vector<float>& pressure, std::string& error) = 0;
    };
}
