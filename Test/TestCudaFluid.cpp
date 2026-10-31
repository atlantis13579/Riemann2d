#include "Test.h"
#include "Fluid/FluidSimulation2D.h"

#include <iostream>

using namespace Riemann2D;

R2_TEST("CUDA liquid selection preserves state or reports unavailable support")
{
    FluidSimulation2D fluid;
    fluid.CreateBlock({-1.0f, 1.0f}, {1.0f, 3.0f}, 0.16f);
    const auto particles = fluid.GetParticles();
    const bool selected = fluid.SetBackend(FluidComputeBackend::CUDA);
    R2_REQUIRE(selected == GetCudaBackendInfo().Available);
    R2_REQUIRE(fluid.GetParticles().size() == particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i)
        R2_REQUIRE((particles[i].Position - fluid.GetParticles()[i].Position).SquareLength() == 0.0f);
    if (!selected) R2_REQUIRE(!fluid.GetBackendError().empty());
    fluid.Step(1.0f / 60.0f);
    R2_REQUIRE(fluid.GetBackend() == (selected ? FluidComputeBackend::CUDA : FluidComputeBackend::CPU));
    R2_REQUIRE(fluid.SetBackend(FluidComputeBackend::CPU));
    fluid.Step(1.0f / 60.0f);
    R2_REQUIRE(fluid.SampleOccupancy({0.0f, 2.0f}) > 0.5f);
}

R2_TEST("CUDA liquid pressure and viscosity agree with CPU for a compressed rotated block")
{
    if (!GetCudaBackendInfo().Available) { std::cout << "[SKIP] CUDA GPU unavailable\n"; return; }
    FluidSimulation2D cpu;
    cpu.GetSettings().ContainerTransform = {{0.4f, 1.0f}, 0.27f};
    cpu.CreateBlock({-1.0f, 1.0f}, {1.0f, 3.0f}, 0.16f);
    cpu.GetSettings().RestKernelDensity = cpu.GetKernelRestDensity() * 0.9f;
    auto gpu = cpu;
    R2_REQUIRE(gpu.SetBackend(FluidComputeBackend::CUDA));
    cpu.Step(1.0f / 60.0f); gpu.Step(1.0f / 60.0f);
    R2_REQUIRE(gpu.GetBackend() == FluidComputeBackend::CUDA);
    for (std::size_t i = 0; i < cpu.GetParticles().size(); ++i)
    {
        R2_REQUIRE((cpu.GetParticles()[i].Position - gpu.GetParticles()[i].Position).Length() < 0.0002f);
        R2_REQUIRE((cpu.GetParticles()[i].Velocity - gpu.GetParticles()[i].Velocity).Length() < 0.03f);
        R2_REQUIRE(std::abs(cpu.GetParticles()[i].Density - gpu.GetParticles()[i].Density) < 0.005f);
    }
    auto copy = gpu;
    copy.GetSettings().Gravity = {0.0f, 0.0f};
    const auto original = gpu.GetParticles();
    copy.Step(1.0f / 60.0f);
    R2_REQUIRE((original[0].Position - gpu.GetParticles()[0].Position).SquareLength() == 0.0f);
    for (int step = 0; step < 180; ++step) gpu.Step(1.0f / 60.0f);
    R2_REQUIRE(gpu.GetBackend() == FluidComputeBackend::CUDA);
    for (const auto& particle : gpu.GetParticles())
    {
        R2_REQUIRE(std::isfinite(particle.Velocity.SquareLength()));
        const auto local = gpu.GetSettings().ContainerTransform.InverseTransformPoint(particle.Position);
        R2_REQUIRE(local.X >= gpu.GetSettings().BoundsMinimum.X - 0.001f);
        R2_REQUIRE(local.Y >= gpu.GetSettings().BoundsMinimum.Y - 0.001f);
        R2_REQUIRE(local.X <= gpu.GetSettings().BoundsMaximum.X + 0.001f);
        R2_REQUIRE(local.Y <= gpu.GetSettings().BoundsMaximum.Y + 0.001f);
    }
}

R2_TEST("CUDA liquid resource failure falls back without losing the current host state")
{
    if (!GetCudaBackendInfo().Available) { std::cout << "[SKIP] CUDA GPU unavailable\n"; return; }
    FluidSimulation2D cpu;
    cpu.CreateBlock({0.0f, 0.0f}, {1.0f, 1.0f}, 0.2f);
    cpu.GetSettings().BoundsMaximum = {100000.0f, 100000.0f};
    auto gpu = cpu;
    gpu.SetBackend(FluidComputeBackend::CUDA);
    gpu.Step(1.0f / 60.0f); cpu.Step(1.0f / 60.0f);
    R2_REQUIRE(gpu.GetBackend() == FluidComputeBackend::CPU);
    R2_REQUIRE(!gpu.GetBackendError().empty());
    for (std::size_t i = 0; i < cpu.GetParticles().size(); ++i)
        R2_REQUIRE((cpu.GetParticles()[i].Position - gpu.GetParticles()[i].Position).SquareLength() == 0.0f);
}
