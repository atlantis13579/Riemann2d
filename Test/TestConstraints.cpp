#include "Test.h"

#include "Collision/Box2.h"
#include "Dynamics/Constraints2D.h"
#include "Dynamics/PhysicsWorld2D.h"

#include <cmath>
#include <memory>
#include <vector>

using namespace Riemann2D;

namespace
{
    std::unique_ptr<RigidBody2D> MakeConstraintBody(BodyType2D type, const Vector2& position)
    {
        RigidBodyDesc2D description;
        description.Type = type;
        description.Shape = std::make_shared<Box2>(Vector2{0.25f, 0.25f});
        description.Transform.Position = position;
        return std::make_unique<RigidBody2D>(description);
    }
}

R2_TEST("Distance constraint converges to its configured length in PhysicsWorld")
{
    PhysicsWorldSettings2D settings;
    settings.Gravity = Vector2::Zero();
    PhysicsWorld2D world(settings);
    RigidBodyDesc2D anchorDescription;
    anchorDescription.Type = BodyType2D::Static;
    anchorDescription.Shape = std::make_shared<Box2>(Vector2{0.2f, 0.2f});
    RigidBody2D& anchor = world.CreateBody(anchorDescription);
    RigidBodyDesc2D bodyDescription;
    bodyDescription.Shape = std::make_shared<Box2>(Vector2{0.2f, 0.2f});
    bodyDescription.Transform.Position = {3.0f, 0.0f};
    RigidBody2D& body = world.CreateBody(bodyDescription);
    DistanceConstraintDesc2D distance;
    distance.BodyA = &anchor;
    distance.BodyB = &body;
    distance.Length = 1.0f;
    world.AddConstraint(std::make_unique<DistanceConstraint2D>(distance));
    for (int step = 0; step < 60; ++step) world.Step(1.0f / 120.0f);
    R2_REQUIRE(NearlyEqual((body.GetTransform().Position - anchor.GetTransform().Position).Length(),
                           1.0f, 0.03f));
}

R2_TEST("Physics world removes a live mouse constraint safely")
{
    PhysicsWorldSettings2D settings;
    settings.Gravity = Vector2::Zero();
    PhysicsWorld2D world(settings);
    RigidBodyDesc2D bodyDescription;
    bodyDescription.Shape = std::make_shared<Box2>(Vector2{0.5f, 0.5f});
    RigidBody2D& body = world.CreateBody(bodyDescription);
    MouseConstraintDesc2D mouseDescription;
    mouseDescription.Body = &body;
    auto mouse = std::make_unique<MouseConstraint2D>(mouseDescription);
    MouseConstraint2D* mousePointer = mouse.get();
    world.AddConstraint(std::move(mouse));

    R2_REQUIRE(world.GetConstraintCount() == 1);
    R2_REQUIRE(world.RemoveConstraint(mousePointer));
    R2_REQUIRE(world.GetConstraintCount() == 0);
    R2_REQUIRE(!world.RemoveConstraint(mousePointer));
    world.Step(1.0f / 60.0f);
}

R2_TEST("Revolute and prismatic constraints remove forbidden position error")
{
    auto anchor = MakeConstraintBody(BodyType2D::Static, Vector2::Zero());
    auto revoluteBody = MakeConstraintBody(BodyType2D::Dynamic, Vector2::Zero());
    RevoluteConstraintDesc2D revoluteDescription;
    revoluteDescription.BodyA = anchor.get();
    revoluteDescription.BodyB = revoluteBody.get();
    RevoluteConstraint2D revolute(revoluteDescription);
    revoluteBody->SetTransform({{1.0f, -0.8f}, 0.0f});
    for (int iteration = 0; iteration < 20; ++iteration) revolute.SolvePosition();
    R2_REQUIRE(revoluteBody->GetTransform().Position.Length() < 0.03f);

    auto sliderBody = MakeConstraintBody(BodyType2D::Dynamic, {0.0f, 2.0f});
    PrismaticConstraintDesc2D prismaticDescription;
    prismaticDescription.BodyA = anchor.get();
    prismaticDescription.BodyB = sliderBody.get();
    prismaticDescription.LocalAxisA = Vector2::UnitX();
    PrismaticConstraint2D prismatic(prismaticDescription);
    for (int iteration = 0; iteration < 20; ++iteration) prismatic.SolvePosition();
    R2_REQUIRE(std::abs(sliderBody->GetTransform().Position.Y) < 0.03f);
}

R2_TEST("All supported constraint types produce finite state")
{
    auto anchor = MakeConstraintBody(BodyType2D::Static, {0.0f, 2.0f});
    auto bodyA = MakeConstraintBody(BodyType2D::Dynamic, {-1.0f, 0.0f});
    auto bodyB = MakeConstraintBody(BodyType2D::Dynamic, {1.0f, 0.0f});

    WeldConstraintDesc2D weldDesc; weldDesc.BodyA = anchor.get(); weldDesc.BodyB = bodyA.get();
    WheelConstraintDesc2D wheelDesc; wheelDesc.BodyA = anchor.get(); wheelDesc.BodyB = bodyA.get();
    MotorConstraintDesc2D motorDesc; motorDesc.BodyA = anchor.get(); motorDesc.BodyB = bodyA.get();
    MouseConstraintDesc2D mouseDesc; mouseDesc.Body = bodyA.get(); mouseDesc.Target = {0.0f, 1.0f};
    PulleyConstraintDesc2D pulleyDesc; pulleyDesc.BodyA = bodyA.get(); pulleyDesc.BodyB = bodyB.get();
    GearConstraintDesc2D gearDesc; gearDesc.BodyA = bodyA.get(); gearDesc.BodyB = bodyB.get();
    FrictionConstraintDesc2D frictionDesc; frictionDesc.BodyA = bodyA.get(); frictionDesc.BodyB = bodyB.get();

    std::vector<std::unique_ptr<IConstraint2D>> constraints;
    constraints.push_back(std::make_unique<WeldConstraint2D>(weldDesc));
    constraints.push_back(std::make_unique<WheelConstraint2D>(wheelDesc));
    constraints.push_back(std::make_unique<MotorConstraint2D>(motorDesc));
    constraints.push_back(std::make_unique<MouseConstraint2D>(mouseDesc));
    constraints.push_back(std::make_unique<PulleyConstraint2D>(pulleyDesc));
    constraints.push_back(std::make_unique<GearConstraint2D>(gearDesc));
    constraints.push_back(std::make_unique<FrictionConstraint2D>(frictionDesc));
    for (const std::unique_ptr<IConstraint2D>& constraint : constraints)
    {
        constraint->SolveVelocity(1.0f / 60.0f);
        constraint->SolvePosition();
    }
    const Transform2& poseA = bodyA->GetTransform();
    const Transform2& poseB = bodyB->GetTransform();
    R2_REQUIRE(std::isfinite(poseA.Position.X) && std::isfinite(poseA.Position.Y));
    R2_REQUIRE(std::isfinite(poseA.Angle));
    R2_REQUIRE(std::isfinite(poseB.Position.X) && std::isfinite(poseB.Position.Y));
    R2_REQUIRE(std::isfinite(poseB.Angle));
}
