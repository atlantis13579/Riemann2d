#include "Test.h"

#include "Collision/Box2.h"
#include "Dynamics/PhysicsWorld2D.h"

#include <limits>

using namespace Riemann2D;

namespace
{
    void AddBox(PhysicsWorld2D& world, Vector2 center, Vector2 halfExtent)
    {
        RigidBodyDesc2D desc;
        desc.Type = BodyType2D::Static;
        desc.Transform.Position = center;
        desc.Shape = std::make_shared<Box2>(halfExtent);
        desc.Restitution = 0.0f;
        world.CreateBody(desc);
    }

    void Step(PhysicsWorld2D& world, int count, float dt = 1.0f / 60.0f)
    {
        for (int i = 0; i < count; ++i) world.Step(dt);
    }
}

R2_TEST("Robot walks and follows the same jump parabola at two time steps")
{
    for (int rate : {30, 120})
    {
        PhysicsWorldSettings2D settings;
        settings.Gravity.Y = -10.0f;
        PhysicsWorld2D world(settings);
        AddBox(world, {0, -0.5f}, {20, 0.5f});
        auto& robot = world.CreateRobotController({}, {0, 2});
        R2_REQUIRE(!robot.Jump(5));
        const float dt = 1.0f / rate;
        Step(world, rate, dt);
        R2_REQUIRE(robot.IsGrounded());
        R2_REQUIRE(!robot.Jump(0) && !robot.Jump(std::numeric_limits<float>::infinity()));
        robot.SetMoveVelocity(-2);
        Step(world, rate / 2, dt);
        R2_REQUIRE(NearlyEqual(robot.GetPosition().X, -1.0f, 0.001f));
        const Vector2 start = robot.GetPosition();
        robot.SetMoveVelocity(2);
        R2_REQUIRE(robot.Jump(5));
        for (int i = 1; i < rate; ++i)
        {
            world.Step(dt);
            const float t = i * dt;
            R2_REQUIRE(NearlyEqual(robot.GetPosition().Y, start.Y + 5 * t - 5 * t * t, 0.001f));
            R2_REQUIRE(NearlyEqual(robot.GetPosition().X, start.X + 2 * t, 0.001f));
            R2_REQUIRE(!robot.IsGrounded() && !robot.Jump(5));
        }
        Step(world, 4, dt);
        R2_REQUIRE(robot.IsGrounded() && robot.Jump(5));
    }
}

R2_TEST("Robot contacts distinguish walls and ceilings from floor support")
{
    PhysicsWorld2D world;
    AddBox(world, {0, -0.5f}, {5, 0.5f});
    AddBox(world, {2, 2}, {0.25f, 2});
    AddBox(world, {0, 2.5f}, {5, 0.1f});
    auto& robot = world.CreateRobotController({}, {0, 0.9f});
    Step(world, 4);
    robot.SetMoveVelocity(4);
    Step(world, 60);
    R2_REQUIRE(NearlyEqual(robot.GetPosition().X, 1.4f, 0.012f));
    R2_REQUIRE(robot.Jump(5));
    float peak = robot.GetPosition().Y;
    for (int i = 0; i < 24; ++i)
    {
        world.Step(1.0f / 60.0f);
        peak = std::max(peak, robot.GetPosition().Y);
        R2_REQUIRE(!robot.Jump(5));
    }
    R2_REQUIRE(peak > 1.45f && peak < 1.54f);
    Step(world, 60);
    R2_REQUIRE(robot.IsGrounded());
}

R2_TEST("Robot loses support when walking off a ledge")
{
    PhysicsWorld2D world;
    auto& robot = world.CreateRobotController({}, {0, 0.9f});
    AddBox(world, {0, -0.5f}, {1, 0.5f});
    Step(world, 10);
    R2_REQUIRE(robot.IsGrounded());
    robot.SetMoveVelocity(3.0f);
    Step(world, 45);
    R2_REQUIRE(robot.GetPosition().X > 2.0f && robot.GetPosition().Y < 0.7f);
    R2_REQUIRE(!robot.IsGrounded() && !robot.Jump(4));
    world.Clear();
    world.Step(1.0f / 60.0f);
    R2_REQUIRE(world.GetBodies().empty());
}
