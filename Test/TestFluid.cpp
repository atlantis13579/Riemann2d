#include "Test.h"
#include "Collision/Box2.h"
#include "Dynamics/PhysicsWorld2D.h"
#include "Fluid/FluidRigidCoupling2D.h"
#include "Fluid/FluidSimulation2D.h"
#include <cmath>
#include <memory>
#include <set>
#include <iostream>

using namespace Riemann2D;

R2_TEST("Fluid spatial grid matches brute force across negative cells and hash collisions")
{
    std::vector<FluidParticle2D> particles;
    for (int i = 0; i < 1200; ++i)
        particles.push_back({{std::sin(static_cast<float>(i) * 1.7f) * 12.0f,
                              std::cos(static_cast<float>(i) * 0.7f) * 12.0f}});
    ParticleGrid2D grid;
    grid.Build(particles, 0.37f);
    for (std::size_t i = 0; i < particles.size(); i += 13)
    {
        std::set<std::size_t> expected;
        std::set<std::size_t> actual;
        constexpr float radius = 0.81f;
        for (std::size_t j = 0; j < particles.size(); ++j)
            if ((particles[j].Position - particles[i].Position).SquareLength() < radius * radius)
                expected.insert(j);
        grid.Query(particles[i].Position, radius, [&](std::size_t j) {
            if ((particles[j].Position - particles[i].Position).SquareLength() < radius * radius)
                R2_REQUIRE(actual.insert(j).second);
        });
        R2_REQUIRE(expected == actual);
    }
}

R2_TEST("Fluid density constraints preserve a resting block without artificial expansion")
{
    FluidSimulation2D fluid;
    fluid.GetSettings().Gravity = Vector2::Zero();
    fluid.CreateBlock({-2.0f, 0.0f}, {2.0f, 4.0f}, 0.16f);
    const auto initial = fluid.GetParticles();
    for (int step = 0; step < 120; ++step) fluid.Step(1.0f / 60.0f);
    for (std::size_t i = 0; i < initial.size(); ++i)
    {
        R2_REQUIRE((fluid.GetParticles()[i].Position - initial[i].Position).Length() < 0.02f);
        R2_REQUIRE(fluid.GetParticles()[i].Density < fluid.GetKernelRestDensity() * 1.03f);
    }
    fluid.Clear();
    R2_REQUIRE(fluid.SampleOccupancy({0.0f, 1.0f}) == 0.0f);
}

R2_TEST("Fluid settled pool remains finite with bounded compression")
{
    FluidSimulation2D fluid;
    fluid.GetSettings().BoundsMinimum = {-2.0f, -1.0f};
    fluid.GetSettings().BoundsMaximum = {2.0f, 5.0f};
    fluid.CreateBlock({-1.8f, 0.0f}, {1.8f, 2.8f}, 0.18f);
    for (int step = 0; step < 360; ++step) fluid.Step(1.0f / 60.0f);
    float meanDensity = 0.0f;
    for (const auto& particle : fluid.GetParticles())
    {
        R2_REQUIRE(std::isfinite(particle.Velocity.SquareLength()));
        R2_REQUIRE(particle.Velocity.Length() < 15.0f);
        meanDensity += particle.Density;
    }
    meanDensity /= static_cast<float>(fluid.GetParticles().size());
    R2_REQUIRE(meanDensity < fluid.GetKernelRestDensity() * 1.15f);
    R2_REQUIRE(meanDensity > fluid.GetKernelRestDensity() * 0.8f);
}

R2_TEST("Fluid particles fall under gravity and remain inside bounds")
{
    FluidSimulation2D fluid;
    fluid.CreateBlock({-1.0f, 3.0f}, {1.0f, 5.0f}, 0.3f);
    const auto centerY = [&fluid]
    {
        float sum = 0.0f;
        for (const FluidParticle2D& particle : fluid.GetParticles()) sum += particle.Position.Y;
        return sum / static_cast<float>(fluid.GetParticles().size());
    };
    const float initialCenter = centerY();
    for (int step = 0; step < 90; ++step) fluid.Step(1.0f / 120.0f);
    R2_REQUIRE(centerY() < initialCenter - 0.5f);
    const FluidSettings2D& settings = fluid.GetSettings();
    for (const FluidParticle2D& particle : fluid.GetParticles())
    {
        R2_REQUIRE(std::isfinite(particle.Position.X) && std::isfinite(particle.Position.Y));
        R2_REQUIRE(std::isfinite(particle.Velocity.X) && std::isfinite(particle.Velocity.Y));
        R2_REQUIRE(particle.Position.X >= settings.BoundsMinimum.X);
        R2_REQUIRE(particle.Position.X <= settings.BoundsMaximum.X);
        R2_REQUIRE(particle.Position.Y >= settings.BoundsMinimum.Y);
        R2_REQUIRE(particle.Position.Y <= settings.BoundsMaximum.Y);
        R2_REQUIRE(particle.Density > 0.0f);
    }
}

R2_TEST("Fluid sampling reports occupancy and local velocity")
{
    FluidSimulation2D fluid;
    fluid.CreateBlock({-1.0f, -1.0f}, {1.0f, 1.0f}, 0.3f);

    R2_REQUIRE(fluid.SampleOccupancy(Vector2::Zero()) > 0.9f);
    R2_REQUIRE(NearlyEqual(fluid.SampleOccupancy({5.0f, 5.0f}), 0.0f));
    R2_REQUIRE(fluid.SampleVelocity(Vector2::Zero()).SquareLength() <= Epsilon);
}

R2_TEST("Rotated fluid container keeps particles sealed in local bounds")
{
    FluidSimulation2D fluid;
    FluidSettings2D& settings = fluid.GetSettings();
    settings.BoundsMinimum = {-1.0f, -1.0f};
    settings.BoundsMaximum = {1.0f, 1.0f};
    settings.ContainerTransform.Position = {2.0f, 3.0f};
    settings.ContainerTransform.Angle = 0.7f;
    fluid.CreateBlock({-0.8f, -0.8f}, {0.8f, 0.8f}, 0.3f);

    for (int step = 0; step < 120; ++step) fluid.Step(1.0f / 120.0f);

    for (const FluidParticle2D& particle : fluid.GetParticles())
    {
        const Vector2 localPosition =
            settings.ContainerTransform.InverseTransformPoint(particle.Position);
        R2_REQUIRE(localPosition.X >= settings.BoundsMinimum.X);
        R2_REQUIRE(localPosition.X <= settings.BoundsMaximum.X);
        R2_REQUIRE(localPosition.Y >= settings.BoundsMinimum.Y);
        R2_REQUIRE(localPosition.Y <= settings.BoundsMaximum.Y);
    }
}

R2_TEST("Fluid force field makes a low-density box buoyant")
{
    FluidSimulation2D fluid;
    fluid.CreateBlock({-1.5f, -1.5f}, {1.5f, 1.5f}, 0.3f);

    PhysicsWorld2D world;
    RigidBodyDesc2D bodyDescription;
    bodyDescription.Shape = std::make_shared<Box2>(Vector2{0.5f, 0.5f});
    bodyDescription.Density = 0.5f;
    bodyDescription.LinearDamping = 0.0f;
    RigidBody2D& body = world.CreateBody(bodyDescription);
    auto coupling = std::make_unique<FluidRigidCoupling2D>(&fluid);
    FluidRigidCouplingResult2D result;
    R2_REQUIRE(coupling->Evaluate(body, result));
    R2_REQUIRE(result.ActiveSampleCount == result.SampleCount);
    R2_REQUIRE(result.BuoyancyForce.Y > 0.0f);
    R2_REQUIRE(result.SubmergedArea > 0.95f);
    world.AddForceField(std::move(coupling));

    world.Step(1.0f / 60.0f);

    R2_REQUIRE(world.GetForceFieldCount() == 1);
    R2_REQUIRE(body.GetLinearVelocity().Y > 0.0f);
}

namespace TestComputeFluidCases
{
using namespace Riemann2D;
#if defined(__APPLE__)
static constexpr FluidComputeBackend NativeBackend = FluidComputeBackend::Metal;
#else
static constexpr FluidComputeBackend NativeBackend = FluidComputeBackend::ComputeShader;
#endif

R2_TEST("Compute Shader liquid selection preserves state or reports unavailable support")
{
    FluidSimulation2D fluid;
    fluid.CreateBlock({-1.0f, 1.0f}, {1.0f, 3.0f}, 0.16f);
    const auto particles = fluid.GetParticles();
    const bool selected = fluid.SetBackend(NativeBackend);
    R2_REQUIRE(selected == GetOperatorAvailability(GpuOperator::Liquid, NativeBackend).Available);
    R2_REQUIRE(fluid.GetParticles().size() == particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i)
        R2_REQUIRE((particles[i].Position - fluid.GetParticles()[i].Position).SquareLength() == 0.0f);
    if (!selected) R2_REQUIRE(!fluid.GetBackendError().empty());
    fluid.Step(1.0f / 60.0f);
    R2_REQUIRE(fluid.GetBackend() == (selected ? NativeBackend : FluidComputeBackend::CPU));
    R2_REQUIRE(fluid.SetBackend(FluidComputeBackend::CPU));
    fluid.Step(1.0f / 60.0f);
    R2_REQUIRE(fluid.SampleOccupancy({0.0f, 2.0f}) > 0.5f);
}

R2_TEST("Compute Shader liquid can switch directly to and from CUDA without resetting particles")
{
    if (!GetOperatorAvailability(GpuOperator::Liquid, NativeBackend).Available || !GetCudaBackendInfo().Available)
    { std::cout << "[SKIP] Both GPU backends are required\n"; return; }
    FluidSimulation2D fluid;
    fluid.CreateBlock({-1.0f, 1.0f}, {1.0f, 3.0f}, 0.16f);
    for (auto backend : {NativeBackend, FluidComputeBackend::CUDA, NativeBackend})
    {
        const auto particles = fluid.GetParticles();
        R2_REQUIRE(fluid.SetBackend(backend));
        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            R2_REQUIRE((particles[i].Position - fluid.GetParticles()[i].Position).SquareLength() == 0.0f);
            R2_REQUIRE((particles[i].Velocity - fluid.GetParticles()[i].Velocity).SquareLength() == 0.0f);
        }
        auto cpu = fluid; cpu.SetBackend(FluidComputeBackend::CPU);
        fluid.Step(1.0f / 120.0f); cpu.Step(1.0f / 120.0f);
        R2_REQUIRE(fluid.GetBackend() == backend);
        for (std::size_t i = 0; i < particles.size(); ++i)
            R2_REQUIRE((cpu.GetParticles()[i].Position - fluid.GetParticles()[i].Position).Length() < 0.0002f);
    }
}

R2_TEST("Compute Shader liquid pressure and viscosity agree with CPU for a compressed rotated block")
{
    if (!GetOperatorAvailability(GpuOperator::Liquid, NativeBackend).Available) { std::cout << "[SKIP] Compute Shader GPU unavailable\n"; return; }
    FluidSimulation2D cpu;
    cpu.GetSettings().ContainerTransform = {{0.4f, 1.0f}, 0.27f};
    cpu.CreateBlock({-1.0f, 1.0f}, {1.0f, 3.0f}, 0.16f);
    cpu.GetSettings().RestKernelDensity = cpu.GetKernelRestDensity() * 0.9f;
    auto gpu = cpu;
    R2_REQUIRE(gpu.SetBackend(NativeBackend));
    cpu.Step(1.0f / 60.0f); gpu.Step(1.0f / 60.0f);
    if (!gpu.GetBackendError().empty()) std::cout << gpu.GetBackendError() << "\n";
    R2_REQUIRE(gpu.GetBackend() == NativeBackend);
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
    if (!gpu.GetBackendError().empty()) std::cout << gpu.GetBackendError() << "\n";
    R2_REQUIRE(gpu.GetBackend() == NativeBackend);
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

R2_TEST("Compute Shader liquid resource failure falls back without losing the current host state")
{
    if (!GetOperatorAvailability(GpuOperator::Liquid, NativeBackend).Available) { std::cout << "[SKIP] Compute Shader GPU unavailable\n"; return; }
    FluidSimulation2D cpu;
    cpu.CreateBlock({0.0f, 0.0f}, {1.0f, 1.0f}, 0.2f);
    cpu.GetSettings().BoundsMaximum = {100000.0f, 100000.0f};
    auto gpu = cpu;
    gpu.SetBackend(NativeBackend);
    gpu.Step(1.0f / 60.0f); cpu.Step(1.0f / 60.0f);
    R2_REQUIRE(gpu.GetBackend() == FluidComputeBackend::CPU);
    R2_REQUIRE(!gpu.GetBackendError().empty());
    for (std::size_t i = 0; i < cpu.GetParticles().size(); ++i)
        R2_REQUIRE((cpu.GetParticles()[i].Position - gpu.GetParticles()[i].Position).SquareLength() == 0.0f);
}
}

namespace TestCudaFluidCases
{
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
}
