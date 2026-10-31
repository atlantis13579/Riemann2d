#include "Test.h"
#include "Fluid/AerodynamicsSimulation2D.h"

#include <iostream>

using namespace Riemann2D;

R2_TEST("Compute Shader smoke selection preserves density and switches without resetting the field")
{
    AerodynamicsSimulation2D air;
    air.Initialize({-3.0f, -3.0f}, {3.0f, 3.0f}, 64, 48);
    air.AddEmitter({0.0f, 0.0f}, 0.7f, 4.0f, {1.0f, 2.0f}, 1.0f / 60.0f);
    const auto density = air.GetDensityField();
    R2_REQUIRE(air.SetBackend(FluidComputeBackend::ComputeShader) == GetComputeShaderBackendInfo().Available);
    R2_REQUIRE(air.GetDensityField() == density);
    if (!GetComputeShaderBackendInfo().Available) R2_REQUIRE(!air.GetBackendError().empty());
    air.Step(1.0f / 60.0f);
    R2_REQUIRE(air.GetBackend() == (GetComputeShaderBackendInfo().Available ? FluidComputeBackend::ComputeShader : FluidComputeBackend::CPU));
    const auto after = air.GetDensityField();
    air.SetBackend(FluidComputeBackend::CPU);
    R2_REQUIRE(air.GetDensityField() == after);
    air.Step(1.0f / 60.0f);
    R2_REQUIRE(air.SampleDensity({0.0f, 0.0f}) > 0.01f);
}

R2_TEST("Compute Shader smoke can switch directly to and from CUDA without resetting the field")
{
    if (!GetComputeShaderBackendInfo().Available || !GetCudaBackendInfo().Available)
    { std::cout << "[SKIP] Both GPU backends are required\n"; return; }
    AerodynamicsSimulation2D air;
    air.Initialize({-3.0f, -3.0f}, {3.0f, 3.0f}, 39, 31);
    for (auto backend : {FluidComputeBackend::ComputeShader, FluidComputeBackend::CUDA, FluidComputeBackend::ComputeShader})
    {
        air.AddEmitter({0.0f, 0.0f}, 0.7f, 4.0f, {1.0f, 2.0f}, 1.0f / 60.0f);
        const auto density = air.GetDensityField();
        R2_REQUIRE(air.SetBackend(backend));
        R2_REQUIRE(air.GetDensityField() == density);
        auto cpu = air; cpu.SetBackend(FluidComputeBackend::CPU);
        air.Step(1.0f / 60.0f); cpu.Step(1.0f / 60.0f);
        R2_REQUIRE(air.GetBackend() == backend);
        for (int y = 0; y < air.GetHeight(); ++y)
            for (int x = 0; x < air.GetWidth(); ++x)
            {
                R2_REQUIRE((air.GetVelocity(x, y) - cpu.GetVelocity(x, y)).Length() < 0.002f);
                R2_REQUIRE(std::abs(air.GetDensity(x, y) - cpu.GetDensity(x, y)) < 0.002f);
            }
    }
}

R2_TEST("Compute Shader smoke projection and transport match CPU on odd and high resolution grids")
{
    if (!GetComputeShaderBackendInfo().Available) { std::cout << "[SKIP] Compute Shader GPU unavailable\n"; return; }
    for (int width : {31, 480})
    {
        AerodynamicsSimulation2D cpu;
        const int height = width == 31 ? 37 : 360;
        cpu.Initialize({-5.0f, -2.0f}, {5.0f, 6.0f}, width, height);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
            {
                const auto p = cpu.GetCellCenter(x, y);
                cpu.AddVelocity(p, {std::sin(p.X) + 0.2f * std::cos(p.Y), std::cos(p.Y)});
                if (p.SquareLength() < 1.0f) cpu.AddDensity(p, 1.0f);
            }
        auto gpu = cpu;
        gpu.SetBackend(FluidComputeBackend::ComputeShader);
        const float before = gpu.ComputeMeanAbsoluteDivergence();
        cpu.ProjectVelocity(); gpu.ProjectVelocity();
        if (!gpu.GetBackendError().empty()) std::cout << gpu.GetBackendError() << "\n";
        R2_REQUIRE(gpu.GetBackend() == FluidComputeBackend::ComputeShader);
        R2_REQUIRE(gpu.ComputeMeanAbsoluteDivergence() < before * 0.05f);
        for (int step = 0; step < 4; ++step)
        {
            const float dt = step % 2 ? 1.0f / 120.0f : 1.0f / 60.0f;
            cpu.GetSettings().UseMacCormack = gpu.GetSettings().UseMacCormack = (step != 2);
            cpu.GetSettings().PressureIterations = gpu.GetSettings().PressureIterations = step < 2 ? 16 : 24;
            cpu.Step(dt); gpu.Step(dt);
            if (!gpu.GetBackendError().empty()) std::cout << gpu.GetBackendError() << '\n';
            R2_REQUIRE(gpu.GetBackend() == FluidComputeBackend::ComputeShader);
        }
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
            {
                R2_REQUIRE((cpu.GetVelocity(x, y) - gpu.GetVelocity(x, y)).Length() < 0.002f);
                R2_REQUIRE(std::abs(cpu.GetDensity(x, y) - gpu.GetDensity(x, y)) < 0.002f);
                R2_REQUIRE(gpu.GetDensity(x, y) >= 0.0f && gpu.GetDensity(x, y) <= 1.00001f);
            }
        R2_REQUIRE(std::abs(gpu.SampleVelocity({-5.0f, 2.0f}).X) < 1.0e-6f);
        R2_REQUIRE(std::abs(gpu.SampleVelocity({0.0f, -2.0f}).Y) < 1.0e-6f);
    }
}

R2_TEST("Compute Shader smoke copies resize safely and preserve constant density without implicit gravity")
{
    if (!GetComputeShaderBackendInfo().Available) { std::cout << "[SKIP] Compute Shader GPU unavailable\n"; return; }
    AerodynamicsSimulation2D air;
    air.SetBackend(FluidComputeBackend::ComputeShader);
    for (int size : {64, 33})
    {
        air.Initialize({-2.0f, -2.0f}, {2.0f, 2.0f}, size, size);
        air.GetSettings().DensityDissipation = 0.0f;
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) air.AddDensity(air.GetCellCenter(x, y), 0.75f);
        air.Step(1.0f / 60.0f);
        auto copy = air;
        copy.AddEmitter({0.0f, 0.0f}, 0.5f, 2.0f, {2.0f, 1.0f}, 0.02f);
        copy.Step(0.02f);
        R2_REQUIRE(copy.GetBackend() == FluidComputeBackend::ComputeShader);
        R2_REQUIRE(air.GetBackend() == FluidComputeBackend::ComputeShader);
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
            {
                R2_REQUIRE(air.GetVelocity(x, y).SquareLength() < 1.0e-12f);
                R2_REQUIRE(std::abs(air.GetDensity(x, y) - 0.75f) < 1.0e-6f);
            }
    }
}
