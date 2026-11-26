#include "Test.h"
#include "Fluid/AerodynamicsSimulation2D.h"
#include <cmath>
#include "Dynamics/PhysicsWorld2D.h"
#include "Collision/Box2.h"
#include "Collision/Capsule2.h"
#include "Collision/Convex2.h"
#include "Collision/Sphere2.h"
#include <limits>
#include <numeric>
#include <iostream>

using namespace Riemann2D;

R2_TEST("Aerodynamics MAC projection removes divergence on odd and rectangular grids")
{
    for (int width : {31, 64, 120})
    {
        AerodynamicsSimulation2D air;
        R2_REQUIRE(air.Initialize({-5.0f, -2.0f}, {5.0f, 2.0f}, width, 37));
        for (int y = 0; y < air.GetHeight(); ++y)
            for (int x = 0; x < air.GetWidth(); ++x)
            {
                const Vector2 p = air.GetCellCenter(x, y);
                air.AddVelocity(p, {std::sin(1.7f * p.X) + 0.3f * std::cos(2.0f * p.Y),
                                    std::cos(1.3f * p.Y) - 0.4f * std::sin(p.X)});
            }
        const float before = air.ComputeMeanAbsoluteDivergence();
        air.ProjectVelocity();
        const float after = air.ComputeMeanAbsoluteDivergence();
        R2_REQUIRE(after < before * 0.05f);
        R2_REQUIRE(std::abs(air.SampleVelocity({-5.0f, 0.0f}).X) < 1.0e-6f);
        R2_REQUIRE(std::abs(air.SampleVelocity({5.0f, 0.0f}).X) < 1.0e-6f);
        R2_REQUIRE(std::abs(air.SampleVelocity({0.0f, -2.0f}).Y) < 1.0e-6f);
        R2_REQUIRE(std::abs(air.SampleVelocity({0.0f, 2.0f}).Y) < 1.0e-6f);
    }
}

R2_TEST("Aerodynamics limited MacCormack preserves constant fields and prevents overshoot")
{
    AerodynamicsSimulation2D air;
    air.Initialize({-3.0f, -3.0f}, {3.0f, 3.0f}, 48, 48);
    air.GetSettings().DensityDissipation = 0.0f;
    for (int y = 0; y < 48; ++y)
        for (int x = 0; x < 48; ++x)
        {
            const auto p = air.GetCellCenter(x, y);
            air.AddDensity(p, 0.75f);
            air.AddVelocity(p, {std::sin(p.Y), std::cos(p.X)});
        }
    for (int step = 0; step < 10; ++step) air.Step(1.0f / 60.0f);
    for (float density : air.GetDensityField()) R2_REQUIRE(std::abs(density - 0.75f) < 1.0e-5f);
    air.Initialize({-3.0f, -3.0f}, {3.0f, 3.0f}, 48, 48);
    for (int y = 0; y < 48; ++y)
        for (int x = 0; x < 48; ++x)
        {
            const auto p = air.GetCellCenter(x, y);
            if (p.SquareLength() < 1.0f) air.AddDensity(p, 1.0f);
            air.AddVelocity(p, {std::sin(p.Y), std::cos(p.X)});
        }
    for (int step = 0; step < 60; ++step) air.Step(1.0f / 60.0f);
    for (float density : air.GetDensityField()) R2_REQUIRE(density >= 0.0f && density <= 1.00001f);
}

R2_TEST("Aerodynamics emitter integrates density consistently across time steps and grids")
{
    float previousMass = 0.0f;
    for (int size : {48, 96})
    {
        AerodynamicsSimulation2D air;
        air.Initialize({-3.0f, -3.0f}, {3.0f, 3.0f}, size, size);
        for (int step = 0; step < size; ++step)
            air.AddEmitter({0.0f, 0.0f}, 0.8f, 3.0f, {2.0f, 0.0f}, 1.0f / size);
        float mass = 0.0f;
        for (float density : air.GetDensityField()) mass += density * 36.0f / (size * size);
        if (previousMass > 0.0f) R2_REQUIRE(std::abs(mass - previousMass) < 0.01f);
        previousMass = mass;
    }
}

R2_TEST("Aerodynamics high resolution projection remains deterministic and convergent")
{
    AerodynamicsSimulation2D air;
    air.Initialize({-7.0f, -2.0f}, {7.0f, 9.0f}, 480, 360);
    air.AddEmitter({-4.0f, 2.0f}, 0.65f, 5.0f, {3.0f, 0.4f}, 1.0f / 60.0f);
    const auto reference = air;
    const float before = air.ComputeMeanAbsoluteDivergence();
    air.ProjectVelocity();
    R2_REQUIRE(air.ComputeMeanAbsoluteDivergence() < before * 0.02f);
    auto duplicate = reference;
    duplicate.ProjectVelocity();
    for (int y = 0; y < 360; y += 11)
        for (int x = 0; x < 480; x += 11)
            R2_REQUIRE((air.GetVelocity(x, y) - duplicate.GetVelocity(x, y)).SquareLength() < 1.0e-12f);
}

R2_TEST("Aerodynamics field has no implicit gravity")
{
    AerodynamicsSimulation2D air;
    R2_REQUIRE(air.Initialize({-4.0f, -3.0f}, {4.0f, 3.0f}, 32, 24));
    for (int step = 0; step < 30; ++step) air.Step(1.0f / 60.0f);
    for (int y = 0; y < air.GetHeight(); ++y)
        for (int x = 0; x < air.GetWidth(); ++x)
            R2_REQUIRE(air.GetVelocity(x, y).SquareLength() < 1.0e-12f);
}

R2_TEST("Aerodynamics field advects density and keeps finite projected velocity")
{
    AerodynamicsSimulation2D air;
    R2_REQUIRE(air.Initialize({-5.0f, -3.0f}, {5.0f, 3.0f}, 40, 24));
    for (int y = 5; y < 19; ++y)
        for (int x = 2; x < 12; ++x)
        {
            const Vector2 center = air.GetCellCenter(x, y);
            air.AddDensity(center, 0.5f);
            air.AddVelocity(center, {2.0f, 0.0f});
        }
    for (int step = 0; step < 20; ++step) air.Step(1.0f / 60.0f);
    float density = 0.0f;
    for (int y = 0; y < air.GetHeight(); ++y)
        for (int x = 0; x < air.GetWidth(); ++x)
        {
            density += air.GetDensity(x, y);
            const Vector2 velocity = air.GetVelocity(x, y);
            R2_REQUIRE(std::isfinite(velocity.X) && std::isfinite(velocity.Y));
        }
    R2_REQUIRE(density > 1.0f);
    R2_REQUIRE(std::isfinite(air.ComputeMeanAbsoluteDivergence()));
    R2_REQUIRE(air.ComputeMeanAbsoluteDivergence() < 1.0f);
}

namespace TestAerodynamicCouplingCases
{
using namespace Riemann2D;

R2_TEST("Pressure field uniform pressure cancels on all supported closed shapes")
{
    const std::vector<std::shared_ptr<const Shape2>> shapes{
        std::make_shared<Sphere2>(0.7f), std::make_shared<Box2>(Vector2{1, 2}),
        std::make_shared<Capsule2>(1.0f, 0.4f),
        std::make_shared<Convex2>(std::vector<Vector2>{{-1,-1},{2,-1},{0,2}})};
    const auto sample = [](const Vector2&) { return AerodynamicSurfaceSample2D{12, 0, {}}; };
    for (const auto& shape : shapes)
    {
        RigidBodyDesc2D desc;
        desc.Shape = shape; desc.Transform = {{3, -2}, 0.65f};
        RigidBody2D body(desc);
        AerodynamicForceResult2D result;
        R2_REQUIRE(AerodynamicsSimulation2D::EvaluateSurfaceForce(body, sample, {}, result));
        R2_REQUIRE(result.Force.Length() < 0.0001f);
        R2_REQUIRE(std::abs(result.Torque) < 0.0001f);
    }
}

R2_TEST("Aerodynamic surface pressure gradient gives displaced-area force and world integration")
{
    PhysicsWorldSettings2D settings; settings.Gravity = {};
    PhysicsWorld2D world(settings);
    const auto sample = [](const Vector2& p)
        { return AerodynamicSurfaceSample2D{10.0f - 2.0f * p.X + p.Y, 0, {}}; };
    for (const auto& shape : std::vector<std::shared_ptr<const Shape2>>{
        std::make_shared<Sphere2>(0.7f), std::make_shared<Box2>(Vector2{0.6f, 1.1f})})
    {
        RigidBodyDesc2D desc; desc.Shape = shape; desc.Density = 2.0f;
        desc.LinearDamping = 0; desc.Transform = {{3.0f * float(world.GetBodies().size()), 0}, 0.7f};
        auto& body = world.CreateBody(desc);
        AerodynamicForceResult2D result;
        R2_REQUIRE(AerodynamicsSimulation2D::EvaluateSurfaceForce(body, sample, {}, result));
        const float area = shape->ComputeMass(1.0f).Mass;
        R2_REQUIRE((result.Force - area * Vector2{2, -1}).Length() < 0.001f);
        R2_REQUIRE(std::abs(result.Torque) < 0.001f);
        body.ApplyForce(result.Force);
    }
    RigidBodyDesc2D fixed; fixed.Type = BodyType2D::Static;
    fixed.Shape = std::make_shared<Sphere2>(0.3f); fixed.Transform.Position = {10, 0};
    auto& staticBody = world.CreateBody(fixed);
    AerodynamicForceResult2D result;
    R2_REQUIRE(!AerodynamicsSimulation2D::EvaluateSurfaceForce(staticBody, sample, {}, result));
    world.Step(1.0f / 60.0f);
    for (const auto& body : world.GetBodies())
        if (body->GetType() == BodyType2D::Dynamic)
            R2_REQUIRE((body->GetLinearVelocity() - Vector2{1, -0.5f} / 60.0f).Length() < 0.0001f);
    R2_REQUIRE(staticBody.GetLinearVelocity().SquareLength() == 0);
    world.Clear();
}

R2_TEST("Pressure field nonuniform surface pressure produces torque")
{
    RigidBodyDesc2D desc; desc.Shape = std::make_shared<Box2>(Vector2{2, 1});
    RigidBody2D body(desc);
    AerodynamicForceSettings2D settings; settings.SurfaceSamples = 192;
    AerodynamicForceResult2D result;
    R2_REQUIRE(AerodynamicsSimulation2D::EvaluateSurfaceForce(body,
        [](const Vector2& p) { return AerodynamicSurfaceSample2D{p.X * p.Y, 0, {}}; }, settings, result));
    R2_REQUIRE(result.Force.Length() < 0.0001f);
    R2_REQUIRE(std::abs(result.Torque + 8.0f) < 0.01f);
}

R2_TEST("Pressure field incident flow pushes bodies and accounts for relative motion")
{
    RigidBodyDesc2D desc; desc.Shape = std::make_shared<Sphere2>(0.5f);
    RigidBody2D body(desc);
    AerodynamicForceSettings2D settings; settings.DragCoefficient = 1.0f;
    const auto sample = [](const Vector2&) { return AerodynamicSurfaceSample2D{0, 1.225f, {3,0}}; };
    AerodynamicForceResult2D result;
    R2_REQUIRE(AerodynamicsSimulation2D::EvaluateSurfaceForce(body, sample, settings, result));
    R2_REQUIRE(result.Force.X > 1 && std::abs(result.Force.Y) < 0.0001f);
    body.SetLinearVelocity({3,0});
    R2_REQUIRE(AerodynamicsSimulation2D::EvaluateSurfaceForce(body, sample, settings, result));
    R2_REQUIRE(result.Force.SquareLength() < 1.e-10f);
    body.SetLinearVelocity({6,0});
    R2_REQUIRE(AerodynamicsSimulation2D::EvaluateSurfaceForce(body, sample, settings, result));
    R2_REQUIRE(result.Force.X < -1);
    desc.Shape = std::make_shared<Box2>(Vector2{2,1});
    RigidBody2D rotating(desc); rotating.SetAngularVelocity(1);
    R2_REQUIRE(AerodynamicsSimulation2D::EvaluateSurfaceForce(rotating,
        [](const Vector2&) { return AerodynamicSurfaceSample2D{0, 1.225f, {}}; }, settings, result));
    R2_REQUIRE(result.Torque < 0);
}

R2_TEST("Pressure field rejects invalid samples and bounds acceleration")
{
    RigidBodyDesc2D desc; desc.Shape = std::make_shared<Sphere2>(0.5f);
    RigidBody2D body(desc);
    AerodynamicForceResult2D result;
    R2_REQUIRE(!AerodynamicsSimulation2D::EvaluateSurfaceForce(body, [](const Vector2&) {
        return AerodynamicSurfaceSample2D{std::numeric_limits<float>::quiet_NaN(), 0, {}};
    }, {}, result));
    R2_REQUIRE(AerodynamicsSimulation2D::EvaluateSurfaceForce(body, [](const Vector2& p) {
        return AerodynamicSurfaceSample2D{-1.e6f * p.X, 0, {}};
    }, {}, result));
    R2_REQUIRE(std::abs(result.Force.Length() * body.GetInverseMass() - 80.0f) < 0.001f);
    R2_REQUIRE(result.ClampScale < 1);
}

R2_TEST("Aerodynamics step applies sampled pressure to registered rigid bodies")
{
    PhysicsWorldSettings2D settings; settings.Gravity = {};
    PhysicsWorld2D world(settings);
    RigidBodyDesc2D desc;
    desc.Shape = std::make_shared<Sphere2>(0.5f);
    desc.LinearDamping = 0.0f;
    desc.AngularDamping = 0.0f;
    auto& body = world.CreateBody(desc);
    AerodynamicsSimulation2D air;
    R2_REQUIRE(air.Initialize({-4,-3}, {4,3}, 64, 48));
    air.SetRigidBodies(&world);
    constexpr float deltaTime = 1.0f / 60.0f;
    air.AddEmitter({-1,0}, 0.8f, 4, {3,0.6f}, deltaTime);
    air.Step(deltaTime);
    AerodynamicForceResult2D result;
    R2_REQUIRE(air.EvaluateBodyForce(body, result));
    R2_REQUIRE(result.SampleCount > 0);
    R2_REQUIRE(result.Force.Length() > 0.0001f);
    world.Step(deltaTime);
    R2_REQUIRE((body.GetLinearVelocity() - deltaTime * body.GetInverseMass() * result.Force).Length() < 0.0001f);
    air.SetRigidBodies(nullptr);
}

R2_TEST("Pressure field smoke projection exposes physical pressure on CPU and GPU")
{
    AerodynamicsSimulation2D base;
    base.Initialize({-4,-3}, {4,3}, 41, 35);
    base.AddEmitter({-1,0}, 0.8f, 4, {3,0.6f}, 1.0f / 60);
    auto reference = base;
    reference.ProjectVelocity(1.0f / 60);
    float maximumPressure = 0;
    for (int y = 0; y < 35; ++y)
        for (int x = 0; x < 41; ++x)
            maximumPressure = std::max(maximumPressure, std::abs(reference.SamplePressure(reference.GetCellCenter(x,y))));
    R2_REQUIRE(maximumPressure > 0.1f);
    auto shorter = base; shorter.ProjectVelocity(1.0f / 120);
    R2_REQUIRE(std::abs(shorter.SamplePressure({-1,0}) - 2 * reference.SamplePressure({-1,0})) < 0.0001f);
    for (auto backend : {FluidComputeBackend::CPU, FluidComputeBackend::CUDA, FluidComputeBackend::ComputeShader, FluidComputeBackend::Metal})
    {
        auto air = base;
        if (!air.SetBackend(backend)) continue;
        air.ProjectVelocity(1.0f / 60);
        R2_REQUIRE(air.GetBackend() == backend);
        for (int y = 1; y < 35; y += 3)
            for (int x = 1; x < 41; x += 3)
            {
                const auto p = air.GetCellCenter(x,y);
                R2_REQUIRE(std::abs(air.SamplePressure(p) - reference.SamplePressure(p)) < 0.01f);
            }
        auto cpu = air; cpu.SetBackend(FluidComputeBackend::CPU);
        cpu.Step(1.0f / 60); air.Step(1.0f / 60);
        R2_REQUIRE(air.GetBackend() == backend);
        R2_REQUIRE(std::abs(air.SamplePressure({-0.7f,0}) - cpu.SamplePressure({-0.7f,0})) < 0.01f);
        R2_REQUIRE(air.SamplePressure({100,100}) == 0);
        air.Clear(); R2_REQUIRE(air.SamplePressure({0,0}) == 0);
    }
}

R2_TEST("Smoke rigid boundary excludes density and conserves displaced smoke")
{
    PhysicsWorld2D world;
    RigidBodyDesc2D desc; desc.Shape = std::make_shared<Sphere2>(0.825f);
    desc.Transform.Position = {-1,0}; desc.GravityScale = 0;
    auto& ball = world.CreateBody(desc);
    for (auto backend : {FluidComputeBackend::CPU, FluidComputeBackend::CUDA, FluidComputeBackend::ComputeShader, FluidComputeBackend::Metal})
    {
        AerodynamicsSimulation2D air;
        air.Initialize({-4,-3},{4,3},80,60);
        if (!air.SetBackend(backend)) continue;
        for (int y = 0; y < 60; ++y)
            for (int x = 0; x < 80; ++x) air.AddDensity(air.GetCellCenter(x,y),0.5f);
        const double mass = std::accumulate(air.GetDensityField().begin(),air.GetDensityField().end(),0.0);
        air.SetRigidBodies(&world);
        R2_REQUIRE(std::abs(std::accumulate(air.GetDensityField().begin(),air.GetDensityField().end(),0.0)-mass) < 0.0001);
        for (int step = 0; step < 8; ++step)
        {
            ball.SetTransform({{-1+0.25f*step,0.1f*step},0});
            ball.SetLinearVelocity({0.4f,0.2f});
            air.SynchronizeRigidBodies();
            air.AddEmitter(ball.GetTransform().Position,0.5f,20,{3,0},1.0f/60);
            air.Step(1.0f/60);
            R2_REQUIRE(air.GetBackend()==backend);
            R2_REQUIRE(air.GetMaximumBoundaryVelocityError() < 1.e-6f);
            for (int y = 0; y < 60; ++y)
                for (int x = 0; x < 80; ++x)
                    if ((air.GetCellCenter(x,y)-ball.GetTransform().Position).Length() <= 0.825f)
                        R2_REQUIRE(air.GetDensityField()[y*80+x] == 0);
            for (int i = 0; i < 24; ++i)
                R2_REQUIRE(air.SampleDensity(ball.GetTransform().Position+
                    0.824f*Vector2{std::cos(i*Pi/12),std::sin(i*Pi/12)}) == 0);
        }
    }
}

R2_TEST("Smoke rigid boundary blocks cross-wall transport on every backend")
{
    PhysicsWorld2D world;
    RigidBodyDesc2D wall; wall.Type = BodyType2D::Static;
    wall.Shape = std::make_shared<Box2>(Vector2{0.12f,4}); world.CreateBody(wall);
    for (auto backend : {FluidComputeBackend::CPU, FluidComputeBackend::CUDA, FluidComputeBackend::ComputeShader, FluidComputeBackend::Metal})
    {
        AerodynamicsSimulation2D air; air.Initialize({-4,-3},{4,3},64,48);
        if (!air.SetBackend(backend)) continue;
        air.SetRigidBodies(&world);
        for (int step = 0; step < 24; ++step)
        {
            air.AddEmitter({-0.65f,0},0.35f,6,{18,0},1.0f/30);
            air.Step(1.0f/30);
            R2_REQUIRE(air.GetBackend()==backend);
            R2_REQUIRE(air.GetMaximumBoundaryVelocityError() < 1.e-6f);
            for (int y = 0; y < 48; ++y)
                for (int x = 32; x < 64; ++x) R2_REQUIRE(air.GetDensityField()[y*64+x] == 0);
        }
        R2_REQUIRE(air.ComputeMeanAbsoluteDivergence() < 0.001f);
    }
}

R2_TEST("Smoke masked projection updates moving obstacles and resized GPU grids")
{
    for (auto backend : {FluidComputeBackend::CPU, FluidComputeBackend::CUDA, FluidComputeBackend::ComputeShader, FluidComputeBackend::Metal})
    {
        PhysicsWorld2D world;
        RigidBodyDesc2D desc; desc.Type = BodyType2D::Static;
        desc.Shape = std::make_shared<Box2>(Vector2{0.4f, 0.7f});
        auto& obstacle = world.CreateBody(desc);
        AerodynamicsSimulation2D air;
        air.GetSettings().PressureIterations = 64;
        if (!air.SetBackend(backend)) continue;
        for (const auto size : {Vector2{65, 49}, Vector2{161, 97}})
        {
            R2_REQUIRE(air.Initialize({-4, -3}, {4, 3}, static_cast<int>(size.X), static_cast<int>(size.Y)));
            for (int phase = 0; phase < 4; ++phase)
            {
                obstacle.SetTransform({{0.3f * phase, 0.1f * phase}, 0.2f * phase});
                air.SetRigidBodies(phase == 2 ? nullptr : &world);
                air.AddEmitter({-1, 0}, 0.8f, 4, {3, 0.6f}, 1.0f / 60);
                air.ProjectVelocity();
                R2_REQUIRE(air.GetBackend() == backend);
                R2_REQUIRE(air.GetBackendError().empty());
                R2_REQUIRE(air.GetMaximumBoundaryVelocityError() < 1.e-6f);
                R2_REQUIRE(air.ComputeMeanAbsoluteDivergence() < 0.001f);
                for (int y = 0; y < air.GetHeight(); ++y)
                    for (int x = 0; x < air.GetWidth(); ++x)
                    {
                        R2_REQUIRE(std::isfinite(air.SamplePressure(air.GetCellCenter(x, y))));
                        R2_REQUIRE(std::isfinite(air.GetDensity(x, y)));
                    }
            }
        }
        // A fully covered domain has no pressure unknowns or density destinations.
        desc.Shape = std::make_shared<Box2>(Vector2{10, 10});
        world.CreateBody(desc);
        air.SynchronizeRigidBodies();
        air.Step(1.0f / 60);
        R2_REQUIRE(air.GetBackend() == backend);
        R2_REQUIRE(air.ComputeMeanAbsoluteDivergence() == 0);
        for (float density : air.GetDensityField()) R2_REQUIRE(density == 0);
    }
}
}

namespace TestComputeAirCases
{
using namespace Riemann2D;
#if defined(__APPLE__)
static constexpr FluidComputeBackend NativeBackend = FluidComputeBackend::Metal;
#else
static constexpr FluidComputeBackend NativeBackend = FluidComputeBackend::ComputeShader;
#endif

R2_TEST("Compute Shader smoke selection preserves density and switches without resetting the field")
{
    AerodynamicsSimulation2D air;
    air.Initialize({-3.0f, -3.0f}, {3.0f, 3.0f}, 64, 48);
    air.AddEmitter({0.0f, 0.0f}, 0.7f, 4.0f, {1.0f, 2.0f}, 1.0f / 60.0f);
    const auto density = air.GetDensityField();
    R2_REQUIRE(air.SetBackend(NativeBackend) == GetOperatorAvailability(GpuOperator::Air, NativeBackend).Available);
    R2_REQUIRE(air.GetDensityField() == density);
    if (!GetOperatorAvailability(GpuOperator::Air, NativeBackend).Available) R2_REQUIRE(!air.GetBackendError().empty());
    air.Step(1.0f / 60.0f);
    R2_REQUIRE(air.GetBackend() == (GetOperatorAvailability(GpuOperator::Air, NativeBackend).Available ? NativeBackend : FluidComputeBackend::CPU));
    const auto after = air.GetDensityField();
    air.SetBackend(FluidComputeBackend::CPU);
    R2_REQUIRE(air.GetDensityField() == after);
    air.Step(1.0f / 60.0f);
    R2_REQUIRE(air.SampleDensity({0.0f, 0.0f}) > 0.01f);
}

R2_TEST("Compute Shader smoke can switch directly to and from CUDA without resetting the field")
{
    if (!GetOperatorAvailability(GpuOperator::Air, NativeBackend).Available || !GetCudaBackendInfo().Available)
    { std::cout << "[SKIP] Both GPU backends are required\n"; return; }
    AerodynamicsSimulation2D air;
    air.Initialize({-3.0f, -3.0f}, {3.0f, 3.0f}, 39, 31);
    for (auto backend : {NativeBackend, FluidComputeBackend::CUDA, NativeBackend})
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
    if (!GetOperatorAvailability(GpuOperator::Air, NativeBackend).Available) { std::cout << "[SKIP] Compute Shader GPU unavailable\n"; return; }
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
        gpu.SetBackend(NativeBackend);
        const float before = gpu.ComputeMeanAbsoluteDivergence();
        cpu.ProjectVelocity(); gpu.ProjectVelocity();
        if (!gpu.GetBackendError().empty()) std::cout << gpu.GetBackendError() << "\n";
        R2_REQUIRE(gpu.GetBackend() == NativeBackend);
        R2_REQUIRE(gpu.ComputeMeanAbsoluteDivergence() < before * 0.05f);
        for (int step = 0; step < 4; ++step)
        {
            const float dt = step % 2 ? 1.0f / 120.0f : 1.0f / 60.0f;
            cpu.GetSettings().UseMacCormack = gpu.GetSettings().UseMacCormack = (step != 2);
            cpu.GetSettings().PressureIterations = gpu.GetSettings().PressureIterations = step < 2 ? 16 : 24;
            cpu.Step(dt); gpu.Step(dt);
            if (!gpu.GetBackendError().empty()) std::cout << gpu.GetBackendError() << '\n';
            R2_REQUIRE(gpu.GetBackend() == NativeBackend);
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
    if (!GetOperatorAvailability(GpuOperator::Air, NativeBackend).Available) { std::cout << "[SKIP] Compute Shader GPU unavailable\n"; return; }
    AerodynamicsSimulation2D air;
    air.SetBackend(NativeBackend);
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
        R2_REQUIRE(copy.GetBackend() == NativeBackend);
        R2_REQUIRE(air.GetBackend() == NativeBackend);
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
            {
                R2_REQUIRE(air.GetVelocity(x, y).SquareLength() < 1.0e-12f);
                R2_REQUIRE(std::abs(air.GetDensity(x, y) - 0.75f) < 1.0e-6f);
            }
    }
}
}

namespace TestCudaAirCases
{
using namespace Riemann2D;

R2_TEST("CUDA smoke selection preserves density and switches without resetting the field")
{
    AerodynamicsSimulation2D air;
    air.Initialize({-3.0f, -3.0f}, {3.0f, 3.0f}, 64, 48);
    air.AddEmitter({0.0f, 0.0f}, 0.7f, 4.0f, {1.0f, 2.0f}, 1.0f / 60.0f);
    const auto density = air.GetDensityField();
    R2_REQUIRE(air.SetBackend(FluidComputeBackend::CUDA) == GetCudaBackendInfo().Available);
    R2_REQUIRE(air.GetDensityField() == density);
    if (!GetCudaBackendInfo().Available) R2_REQUIRE(!air.GetBackendError().empty());
    air.Step(1.0f / 60.0f);
    R2_REQUIRE(air.GetBackend() == (GetCudaBackendInfo().Available ? FluidComputeBackend::CUDA : FluidComputeBackend::CPU));
    const auto after = air.GetDensityField();
    air.SetBackend(FluidComputeBackend::CPU);
    R2_REQUIRE(air.GetDensityField() == after);
    air.Step(1.0f / 60.0f);
    R2_REQUIRE(air.SampleDensity({0.0f, 0.0f}) > 0.01f);
}

R2_TEST("CUDA smoke projection and transport match CPU on odd and high resolution grids")
{
    if (!GetCudaBackendInfo().Available) { std::cout << "[SKIP] CUDA GPU unavailable\n"; return; }
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
        gpu.SetBackend(FluidComputeBackend::CUDA);
        const float before = gpu.ComputeMeanAbsoluteDivergence();
        cpu.ProjectVelocity(); gpu.ProjectVelocity();
        R2_REQUIRE(gpu.GetBackend() == FluidComputeBackend::CUDA);
        R2_REQUIRE(gpu.ComputeMeanAbsoluteDivergence() < before * 0.05f);
        for (int step = 0; step < 4; ++step)
        {
            const float dt = step % 2 ? 1.0f / 120.0f : 1.0f / 60.0f;
            cpu.GetSettings().UseMacCormack = gpu.GetSettings().UseMacCormack = (step != 2);
            cpu.GetSettings().PressureIterations = gpu.GetSettings().PressureIterations = step < 2 ? 16 : 24;
            cpu.Step(dt); gpu.Step(dt);
            if (!gpu.GetBackendError().empty()) std::cout << gpu.GetBackendError() << '\n';
            R2_REQUIRE(gpu.GetBackend() == FluidComputeBackend::CUDA);
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

R2_TEST("CUDA smoke copies resize safely and preserve constant density without implicit gravity")
{
    if (!GetCudaBackendInfo().Available) { std::cout << "[SKIP] CUDA GPU unavailable\n"; return; }
    AerodynamicsSimulation2D air;
    air.SetBackend(FluidComputeBackend::CUDA);
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
        R2_REQUIRE(copy.GetBackend() == FluidComputeBackend::CUDA);
        R2_REQUIRE(air.GetBackend() == FluidComputeBackend::CUDA);
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
            {
                R2_REQUIRE(air.GetVelocity(x, y).SquareLength() < 1.0e-12f);
                R2_REQUIRE(std::abs(air.GetDensity(x, y) - 0.75f) < 1.0e-6f);
            }
    }
}
}
