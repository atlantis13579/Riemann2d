#include "Test.h"

#include "Dynamics/PhysicsWorld2D.h"
#include "Collision/Box2.h"
#include "Collision/Capsule2.h"
#include "Collision/Convex2.h"
#include "Collision/Sphere2.h"
#include "Fluid/AerodynamicsSimulation2D.h"

#include <cmath>
#include <limits>
#include <numeric>

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
    for (auto backend : {FluidComputeBackend::CPU, FluidComputeBackend::CUDA, FluidComputeBackend::ComputeShader})
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
    for (auto backend : {FluidComputeBackend::CPU, FluidComputeBackend::CUDA, FluidComputeBackend::ComputeShader})
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
    for (auto backend : {FluidComputeBackend::CPU, FluidComputeBackend::CUDA, FluidComputeBackend::ComputeShader})
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
    for (auto backend : {FluidComputeBackend::CPU, FluidComputeBackend::CUDA, FluidComputeBackend::ComputeShader})
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
