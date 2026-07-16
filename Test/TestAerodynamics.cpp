#include "Test.h"

#include "Fluid/AerodynamicsSimulation2D.h"

#include <cmath>

using namespace Riemann2D;

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
