#include "Test.h"
#include "Fluid/FluidSimulation2D.h"
#include "Fluid/AerodynamicsSimulation2D.h"
#include "Continuum/SnowSimulation2D.h"
#include <chrono>
#include <cstdio>

using namespace Riemann2D;

R2_TEST("Build and device queries gate compiled operators without running solvers")
{
    const auto start = std::chrono::steady_clock::now();
    ProbeGpuOperators();
    std::printf("Build/device checks completed: %.3f ms\n", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count());
    for (auto backend : {FluidComputeBackend::CPU, FluidComputeBackend::CUDA, FluidComputeBackend::ComputeShader,
                         FluidComputeBackend::Metal})
    {
        FluidSimulation2D liquid;
        AerodynamicsSimulation2D air;
        const auto &liquidInfo = GetOperatorAvailability(GpuOperator::Liquid, backend);
        const auto &airInfo = GetOperatorAvailability(GpuOperator::Air, backend);
        const bool deviceAvailable = backend == FluidComputeBackend::CPU ? true
            : backend == FluidComputeBackend::CUDA ? GetCudaBackendInfo().Available
            : backend == FluidComputeBackend::ComputeShader ? GetComputeShaderBackendInfo().Available
            : GetMetalBackendInfo().Available;
        R2_REQUIRE(liquidInfo.Available == deviceAvailable);
        R2_REQUIRE(airInfo.Available == deviceAvailable);
        for (auto operation : {GpuOperator::Liquid, GpuOperator::Air, GpuOperator::Snow})
            R2_REQUIRE(GetOperatorAvailability(operation, backend).Available == deviceAvailable);
        R2_REQUIRE(liquid.SetBackend(backend) == liquidInfo.Available);
        R2_REQUIRE(air.SetBackend(backend) == airInfo.Available);
        if (!liquidInfo.Available)
            R2_REQUIRE(liquid.GetBackend() == FluidComputeBackend::CPU);
        if (!airInfo.Available)
            R2_REQUIRE(air.GetBackend() == FluidComputeBackend::CPU);
        SnowSimulation2D snow;
        const auto snowBackend = backend == FluidComputeBackend::CPU     ? ContinuumBackend::CPU
                                 : backend == FluidComputeBackend::CUDA  ? ContinuumBackend::CUDA
                                 : backend == FluidComputeBackend::Metal ? ContinuumBackend::Metal
                                                                         : ContinuumBackend::ComputeShader;
        R2_REQUIRE(snow.SetBackend(snowBackend) == GetOperatorAvailability(GpuOperator::Snow, backend).Available);
    }
    R2_REQUIRE(!GetOperatorAvailability(GpuOperator::Liquid, static_cast<FluidComputeBackend>(255)).Available);
    FluidSimulation2D liquid;
    R2_REQUIRE(!liquid.SetBackend(static_cast<FluidComputeBackend>(255)));
}
