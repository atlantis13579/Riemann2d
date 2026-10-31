#pragma once

#include "Fluid/GpuFluidSolvers.h"

namespace Riemann2D
{
    using CudaFluidSolver = FluidGpuSolver;
    using CudaAirSolver = AirGpuSolver;
    std::unique_ptr<CudaFluidSolver> CreateCudaFluidSolver();
    std::unique_ptr<CudaAirSolver> CreateCudaAirSolver();
}
