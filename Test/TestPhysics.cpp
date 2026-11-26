#include "Test.h"
#include "Collision/Box2.h"
#include "Collision/Capsule2.h"
#include "Collision/Convex2.h"
#include "Collision/Sphere2.h"
#include "Collision/TriangleMesh2.h"
#include "Dynamics/PhysicsWorld2D.h"
#include "Maths/Maths.h"
#include <memory>
#include "Dynamics/Constraints2D.h"
#include <cmath>
#include <vector>
#include "Collision/Contact2.h"
#include "Dynamics/ContactSolver2D.h"
#include "Dynamics/LinearAlgebraBackend2D.h"

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

namespace TestConstraintsCases
{
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
}

namespace TestSolversCases
{
using namespace Riemann2D;

namespace
{
    struct SolverRun
    {
        Vector2 Velocity;
        float Residual = 0.0f;
    };

    SolverRun RunSingleContact(ContactSolverType2D type, SolverBackend2D backend)
    {
        const auto shape = std::make_shared<Box2>(Vector2{1.0f, 1.0f});
        RigidBodyDesc2D groundDescription;
        groundDescription.Type = BodyType2D::Static;
        groundDescription.Shape = shape;
        RigidBody2D ground(groundDescription);

        RigidBodyDesc2D bodyDescription;
        bodyDescription.Shape = shape;
        bodyDescription.Transform.Position = {0.0f, 1.8f};
        bodyDescription.LinearVelocity = {1.5f, -3.0f};
        RigidBody2D body(bodyDescription);
        Contact2 contact;
        R2_REQUIRE(GenerateContact(ground, body, contact));
        std::vector<Contact2> contacts{contact};
        ContactSolver2D solver;
        solver.Solve(contacts, type, backend, 8);
        return {body.GetLinearVelocity(), solver.GetLastResidual()};
    }
}

R2_TEST("Scalar and SIMD dense linear algebra backends agree")
{
    constexpr int size = 7;
    float matrix[size * size]{};
    float vector[size]{};
    for (int row = 0; row < size; ++row)
    {
        vector[row] = 0.25f * static_cast<float>(row + 1);
        for (int column = 0; column < size; ++column)
        {
            matrix[row * size + column] =
                static_cast<float>(((row + 3) * (column + 5)) % 11 - 5) * 0.2f;
        }
    }
    float scalar[size]{};
    float simd[size]{};
    DenseMatrixVectorMultiply(matrix, vector, scalar, size, SolverBackend2D::ScalarCPU);
    DenseMatrixVectorMultiply(matrix, vector, simd, size, SolverBackend2D::SIMDCPU);
    for (int index = 0; index < size; ++index)
    {
        R2_REQUIRE(NearlyEqual(scalar[index], simd[index], 1.0e-5f));
    }
    R2_REQUIRE(NearlyEqual(Dot(vector, vector, size, SolverBackend2D::ScalarCPU),
                           Dot(vector, vector, size, SolverBackend2D::SIMDCPU), 1.0e-5f));
}

R2_TEST("Sequential impulse scalar and SIMD selections produce matching contact response")
{
    const SolverRun scalar = RunSingleContact(ContactSolverType2D::SequentialImpulse,
                                               SolverBackend2D::ScalarCPU);
    const SolverRun simd = RunSingleContact(ContactSolverType2D::SequentialImpulse,
                                             SolverBackend2D::SIMDCPU);
    R2_REQUIRE(scalar.Velocity.Y >= -1.0e-4f);
    R2_REQUIRE((scalar.Velocity - simd.Velocity).Length() < 1.0e-5f);
}

R2_TEST("LCP solver satisfies complementarity with scalar and SIMD backends")
{
    const SolverRun scalar = RunSingleContact(ContactSolverType2D::LCP,
                                               SolverBackend2D::ScalarCPU);
    const SolverRun simd = RunSingleContact(ContactSolverType2D::LCP,
                                             SolverBackend2D::SIMDCPU);
    R2_REQUIRE(scalar.Velocity.Y >= -1.0e-4f);
    R2_REQUIRE(simd.Velocity.Y >= -1.0e-4f);
    R2_REQUIRE(scalar.Residual < 1.0e-4f);
    R2_REQUIRE(simd.Residual < 1.0e-4f);
    R2_REQUIRE((scalar.Velocity - simd.Velocity).Length() < 1.0e-4f);
}

R2_TEST("Physics world can step with the LCP SIMD solver")
{
    PhysicsWorldSettings2D settings;
    settings.SolverType = ContactSolverType2D::LCP;
    settings.SolverBackend = SolverBackend2D::SIMDCPU;
    PhysicsWorld2D world(settings);
    RigidBodyDesc2D ground;
    ground.Type = BodyType2D::Static;
    ground.Shape = std::make_shared<Box2>(Vector2{5.0f, 0.5f});
    ground.Transform.Position = {0.0f, -0.5f};
    world.CreateBody(ground);
    RigidBodyDesc2D box;
    box.Shape = std::make_shared<Box2>(Vector2{0.5f, 0.5f});
    box.Transform.Position = {0.0f, 2.0f};
    RigidBody2D& body = world.CreateBody(box);
    for (int step = 0; step < 180; ++step) world.Step(1.0f / 120.0f);
    R2_REQUIRE(body.GetTransform().Position.Y > 0.4f);
    R2_REQUIRE(body.GetTransform().Position.Y < 0.7f);
}
}
