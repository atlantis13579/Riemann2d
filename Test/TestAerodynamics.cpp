#include "Test.h"

#include "Fluid/AerodynamicsSimulation2D.h"

#include <cmath>

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
