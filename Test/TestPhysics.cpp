#include "Test.h"

#include "Collision/Box2.h"
#include "Collision/Capsule2.h"
#include "Collision/Convex2.h"
#include "Collision/Sphere2.h"
#include "Collision/TriangleMesh2.h"
#include "Dynamics/PhysicsWorld2D.h"
#include "Maths/Maths.h"

#include <memory>

using namespace Riemann2D;

R2_TEST("Primitive shapes compute planar mass properties")
{
    const Box2 box({2.0f, 1.0f});
    const MassProperties2 boxMass = box.ComputeMass(3.0f);
    R2_REQUIRE(NearlyEqual(boxMass.Mass, 24.0f));
    R2_REQUIRE(NearlyEqual(boxMass.Inertia, 40.0f));

    const Sphere2 sphere(2.0f);
    const MassProperties2 sphereMass = sphere.ComputeMass(1.0f);
    R2_REQUIRE(NearlyEqual(sphereMass.Mass, 4.0f * Pi));
    R2_REQUIRE(NearlyEqual(sphereMass.Inertia, 8.0f * Pi));
}

R2_TEST("Physics world integrates a dynamic rigid body")
{
    PhysicsWorld2D world;
    RigidBodyDesc2D bodyDesc;
    bodyDesc.Shape = std::make_shared<Sphere2>(0.5f);
    bodyDesc.Transform.Position = {0.0f, 5.0f};
    RigidBody2D& body = world.CreateBody(bodyDesc);
    world.Step(1.0f / 60.0f);
    R2_REQUIRE(body.GetTransform().Position.Y < 5.0f);
    R2_REQUIRE(body.GetLinearVelocity().Y < 0.0f);
}

R2_TEST("Physics world integrates rigid body orientation")
{
    PhysicsWorldSettings2D settings;
    settings.Gravity = Vector2::Zero();
    PhysicsWorld2D world(settings);
    RigidBodyDesc2D bodyDesc;
    bodyDesc.Shape = std::make_shared<Box2>(Vector2{0.5f, 0.5f});
    bodyDesc.AngularVelocity = 2.0f;
    bodyDesc.AngularDamping = 0.0f;
    RigidBody2D& body = world.CreateBody(bodyDesc);

    world.Step(1.0f / 60.0f);

    R2_REQUIRE(NearlyEqual(body.GetTransform().Angle, 1.0f / 30.0f, 1.0e-5f));
}

R2_TEST("Shape gallery primitives fall while triangle mesh remains static")
{
    PhysicsWorld2D world;
    RigidBodyDesc2D mesh;
    mesh.Type = BodyType2D::Static;
    mesh.Shape = std::make_shared<TriangleMesh2>(
        std::vector<Vector2>{{-5.0f, -0.5f}, {5.0f, -0.5f},
                             {5.0f, 0.5f}, {-5.0f, 0.5f}},
        std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3});
    RigidBody2D& meshBody = world.CreateBody(mesh);

    RigidBodyDesc2D body;
    body.Transform.Position = {-3.0f, 4.0f};
    body.Shape = std::make_shared<Box2>(Vector2{0.5f, 0.5f});
    RigidBody2D* box = &world.CreateBody(body);
    body.Transform.Position = {-1.0f, 4.0f};
    body.Shape = std::make_shared<Sphere2>(0.5f);
    RigidBody2D* sphere = &world.CreateBody(body);
    body.Transform.Position = {1.0f, 4.0f};
    body.Shape = std::make_shared<Capsule2>(0.5f, 0.3f);
    RigidBody2D* capsule = &world.CreateBody(body);
    body.Transform.Position = {3.0f, 4.0f};
    body.Shape = std::make_shared<Convex2>(std::vector<Vector2>{
        {-0.5f, -0.4f}, {0.5f, -0.4f}, {0.0f, 0.6f}});
    RigidBody2D* convex = &world.CreateBody(body);

    world.Step(1.0f / 60.0f);

    R2_REQUIRE(NearlyEqual(meshBody.GetTransform().Position.Y, 0.0f));
    R2_REQUIRE(box->GetTransform().Position.Y < 4.0f);
    R2_REQUIRE(sphere->GetTransform().Position.Y < 4.0f);
    R2_REQUIRE(capsule->GetTransform().Position.Y < 4.0f);
    R2_REQUIRE(convex->GetTransform().Position.Y < 4.0f);
}

R2_TEST("Tilted falling box receives angular impulse at ground contact")
{
    PhysicsWorld2D world;
    RigidBodyDesc2D ground;
    ground.Type = BodyType2D::Static;
    ground.Shape = std::make_shared<Box2>(Vector2{5.0f, 0.5f});
    ground.Transform.Position = {0.0f, -0.5f};
    world.CreateBody(ground);

    RigidBodyDesc2D falling;
    falling.Shape = std::make_shared<Box2>(Vector2{1.0f, 1.0f});
    falling.Transform.Position = {0.0f, 3.0f};
    falling.Transform.Angle = 0.18f;
    RigidBody2D& body = world.CreateBody(falling);

    float maximumAngularSpeed = 0.0f;
    for (int step = 0; step < 180; ++step)
    {
        world.Step(1.0f / 120.0f);
        maximumAngularSpeed = std::max(maximumAngularSpeed,
                                       std::abs(body.GetAngularVelocity()));
    }

    R2_REQUIRE(maximumAngularSpeed > 0.05f);
}

R2_TEST("Dynamic box settles on a static box")
{
    PhysicsWorld2D world;
    const auto groundShape = std::make_shared<Box2>(Vector2{5.0f, 0.5f});
    RigidBodyDesc2D ground;
    ground.Type = BodyType2D::Static;
    ground.Shape = groundShape;
    ground.Transform.Position = {0.0f, -0.5f};
    world.CreateBody(ground);

    RigidBodyDesc2D falling;
    falling.Shape = std::make_shared<Box2>(Vector2{0.5f, 0.5f});
    falling.Transform.Position = {0.0f, 3.0f};
    RigidBody2D& body = world.CreateBody(falling);
    for (int step = 0; step < 240; ++step)
    {
        world.Step(1.0f / 120.0f);
    }
    R2_REQUIRE(body.GetTransform().Position.Y > 0.42f);
    R2_REQUIRE(body.GetTransform().Position.Y < 0.65f);
    R2_REQUIRE(std::abs(body.GetLinearVelocity().Y) < 0.2f);
}
