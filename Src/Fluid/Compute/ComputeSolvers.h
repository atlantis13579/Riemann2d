#pragma once

#include "Fluid/GpuFluidSolvers.h"

namespace Riemann2D
{
    std::unique_ptr<FluidGpuSolver> CreateComputeFluidSolver();
    std::unique_ptr<AirGpuSolver> CreateComputeAirSolver();
}
