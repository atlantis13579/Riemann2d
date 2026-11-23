#pragma once

#include "FluidComputeBackend.h"

#include <vector>

namespace Riemann2D
{
    struct FluidParticle2D;
    struct FluidSettings2D;
    struct AerodynamicsSettings2D;
    struct Vector2;

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
                          int width, int height, float deltaTime, bool projectOnly,
                          std::vector<float>& density, std::vector<float>& u,
                          std::vector<float>& v, std::vector<float>& pressure, std::string& error) = 0;
    };
}
