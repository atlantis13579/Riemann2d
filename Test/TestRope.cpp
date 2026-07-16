#include "Test.h"

#include "SoftBody/RopeSimulation2D.h"

#include <algorithm>

using namespace Riemann2D;

R2_TEST("XPBD rope keeps pinned endpoints and limits stretch")
{
    RopeSimulation2D rope;
    rope.CreateLine({-5.0f, 4.0f}, {5.0f, 4.0f}, 21);
    rope.PinParticle(0, {-5.0f, 4.0f});
    rope.PinParticle(20, {5.0f, 4.0f});
    for (int step = 0; step < 240; ++step) rope.Step(1.0f / 120.0f);
    const auto& particles = rope.GetParticles();
    R2_REQUIRE((particles.front().Position - Vector2{-5.0f, 4.0f}).Length() < 1.0e-6f);
    R2_REQUIRE((particles.back().Position - Vector2{5.0f, 4.0f}).Length() < 1.0e-6f);
    R2_REQUIRE(particles[10].Position.Y < 3.9f);
    float maximumLength = 0.0f;
    for (std::size_t index = 0; index + 1 < particles.size(); ++index)
    {
        maximumLength = std::max(maximumLength,
            (particles[index + 1].Position - particles[index].Position).Length());
    }
    R2_REQUIRE(maximumLength < 0.56f);
}

R2_TEST("Pinned rope anchor can be moved without velocity spikes")
{
    RopeSimulation2D rope;
    rope.CreateLine({0.0f, 2.0f}, {4.0f, 2.0f}, 9);
    rope.PinParticle(0, {0.0f, 2.0f});
    rope.SetParticlePosition(0, {0.0f, 3.0f});
    rope.Step(1.0f / 60.0f);
    R2_REQUIRE((rope.GetParticles()[0].Position - Vector2{0.0f, 3.0f}).Length() < 1.0e-6f);
    R2_REQUIRE(rope.GetParticles()[0].Velocity.Length() < 1.0e-6f);
}
