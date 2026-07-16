#include "Test.h"

#include "Collision/Box2.h"
#include "Dynamics/Ragdoll2D.h"

#include <cmath>
#include <memory>

using namespace Riemann2D;

R2_TEST("Ragdoll builds articulated parts and remains finite while falling")
{
    PhysicsWorld2D world;
    RigidBodyDesc2D ground;
    ground.Type = BodyType2D::Static;
    ground.Shape = std::make_shared<Box2>(Vector2{8.0f, 0.4f});
    ground.Transform.Position = {0.0f, -1.0f};
    world.CreateBody(ground);
    Ragdoll2D ragdoll;
    R2_REQUIRE(ragdoll.Build(world, {0.0f, 5.0f}, 0.8f));
    R2_REQUIRE(ragdoll.GetParts().size() == 10);
    for (int step = 0; step < 180; ++step) world.Step(1.0f / 120.0f);
    for (const RagdollPart2D& part : ragdoll.GetParts())
    {
        const Transform2& pose = part.Body->GetTransform();
        R2_REQUIRE(std::isfinite(pose.Position.X) && std::isfinite(pose.Position.Y));
        R2_REQUIRE(std::isfinite(pose.Angle));
        R2_REQUIRE(pose.Position.Y > -3.0f);
    }
}
