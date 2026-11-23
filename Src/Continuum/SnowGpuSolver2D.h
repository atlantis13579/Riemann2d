#pragma once

#include "SnowSimulation2D.h"

namespace Riemann2D
{
    class SnowGpuSolver2D
    {
    public:
        virtual ~SnowGpuSolver2D() = default;
        virtual bool Step(const SnowSettings2D& settings, const SnowCapsule2D& capsule,
                          float deltaTime, std::vector<SnowParticle2D>& particles, std::string& error) = 0;
    };

    std::unique_ptr<SnowGpuSolver2D> CreateCudaSnowSolver2D();
    std::unique_ptr<SnowGpuSolver2D> CreateComputeSnowSolver2D();
}
