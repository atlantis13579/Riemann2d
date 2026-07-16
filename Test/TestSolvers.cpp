#include "Test.h"

#include "Collision/Box2.h"
#include "Collision/Contact2.h"
#include "Dynamics/PhysicsWorld2D.h"
#include "Dynamics/ContactSolver2D.h"
#include "Dynamics/LinearAlgebraBackend2D.h"

#include <memory>
#include <vector>

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
