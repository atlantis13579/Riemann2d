#include "Test.h"

#include "Collision/Box2.h"
#include "Collision/Capsule2.h"
#include "Collision/Contact2.h"
#include "Collision/Convex2.h"
#include "Collision/Sphere2.h"
#include "Collision/TriangleMesh2.h"
#include "Dynamics/RigidBody2D.h"
#include "Maths/Maths.h"

#include <memory>
#include <vector>

using namespace Riemann2D;

namespace
{
    std::unique_ptr<RigidBody2D> MakeStaticBody(std::shared_ptr<const Shape2> shape,
                                                const Vector2& position)
    {
        RigidBodyDesc2D description;
        description.Type = BodyType2D::Static;
        description.Shape = std::move(shape);
        description.Transform.Position = position;
        return std::make_unique<RigidBody2D>(description);
    }

    void CheckPrimitivePair(const std::shared_ptr<const Shape2>& shapeA,
                            const std::shared_ptr<const Shape2>& shapeB,
                            float overlapDistance, float separatedDistance)
    {
        std::unique_ptr<RigidBody2D> bodyA = MakeStaticBody(shapeA, Vector2::Zero());
        std::unique_ptr<RigidBody2D> bodyB = MakeStaticBody(shapeB, {overlapDistance, 0.0f});
        Contact2 contactAB;
        Contact2 contactBA;
        R2_REQUIRE(GenerateContact(*bodyA, *bodyB, contactAB));
        R2_REQUIRE(GenerateContact(*bodyB, *bodyA, contactBA));
        R2_REQUIRE(contactAB.BodyA == bodyA.get() && contactAB.BodyB == bodyB.get());
        R2_REQUIRE(contactBA.BodyA == bodyB.get() && contactBA.BodyB == bodyA.get());
        R2_REQUIRE(contactAB.Normal.X > 0.95f);
        R2_REQUIRE(contactBA.Normal.X < -0.95f);
        R2_REQUIRE((contactAB.Normal + contactBA.Normal).Length() < 0.05f);
        R2_REQUIRE(contactAB.Penetration > 0.0f);
        R2_REQUIRE(NearlyEqual(contactAB.Penetration, contactBA.Penetration, 2.0e-3f));

        bodyB->SetTransform({{separatedDistance, 0.0f}, 0.0f});
        R2_REQUIRE(!GenerateContact(*bodyA, *bodyB, contactAB));
        R2_REQUIRE(!GenerateContact(*bodyB, *bodyA, contactBA));
    }

    std::shared_ptr<const TriangleMesh2> MakeWallMesh()
    {
        return std::make_shared<TriangleMesh2>(
            std::vector<Vector2>{{-1.0f, 8.0f}, {2.0f, 8.0f}, {-1.0f, 11.0f},
                                 {-1.0f, -2.0f}, {2.0f, -2.0f}, {-1.0f, 2.0f}},
            std::vector<std::uint32_t>{0, 1, 2, 3, 4, 5});
    }

    void CheckPrimitiveMesh(const std::shared_ptr<const Shape2>& primitive,
                            float overlapX, float separatedX)
    {
        std::unique_ptr<RigidBody2D> bodyA = MakeStaticBody(primitive, {overlapX, 0.0f});
        std::unique_ptr<RigidBody2D> bodyB = MakeStaticBody(MakeWallMesh(), {2.0f, 0.0f});
        Contact2 contactAB;
        Contact2 contactBA;
        R2_REQUIRE(GenerateContact(*bodyA, *bodyB, contactAB));
        R2_REQUIRE(GenerateContact(*bodyB, *bodyA, contactBA));
        R2_REQUIRE(contactAB.Normal.X > 0.9f);
        R2_REQUIRE(contactBA.Normal.X < -0.9f);
        R2_REQUIRE(contactAB.TriangleIndexA == -1 && contactAB.TriangleIndexB == 1);
        R2_REQUIRE(contactBA.TriangleIndexA == 1 && contactBA.TriangleIndexB == -1);
        R2_REQUIRE(NearlyEqual(contactAB.Penetration, contactBA.Penetration, 2.0e-3f));

        bodyA->SetTransform({{separatedX, 0.0f}, 0.0f});
        R2_REQUIRE(!GenerateContact(*bodyA, *bodyB, contactAB));
        R2_REQUIRE(!GenerateContact(*bodyB, *bodyA, contactBA));
    }
}

R2_TEST("All ten convex primitive collision pairs are dispatched symmetrically")
{
    const auto box = std::make_shared<Box2>(Vector2{1.0f, 1.0f});
    const auto sphere = std::make_shared<Sphere2>(0.75f);
    const auto capsule = std::make_shared<Capsule2>(0.75f, 0.5f);
    const auto convex = std::make_shared<Convex2>(
        std::vector<Vector2>{{-0.75f, -0.75f}, {0.75f, -0.75f},
                             {0.75f, 0.75f}, {-0.75f, 0.75f}});

    CheckPrimitivePair(box, box, 1.75f, 2.25f);
    CheckPrimitivePair(box, sphere, 1.50f, 2.00f);
    CheckPrimitivePair(box, capsule, 2.00f, 2.50f);
    CheckPrimitivePair(box, convex, 1.50f, 2.00f);
    CheckPrimitivePair(sphere, sphere, 1.25f, 1.75f);
    CheckPrimitivePair(sphere, capsule, 1.75f, 2.25f);
    CheckPrimitivePair(sphere, convex, 1.25f, 1.75f);
    CheckPrimitivePair(capsule, capsule, 2.25f, 2.75f);
    CheckPrimitivePair(capsule, convex, 1.75f, 2.25f);
    CheckPrimitivePair(convex, convex, 1.25f, 1.75f);
}

R2_TEST("All four primitive to triangle mesh pairs report child triangle identity")
{
    CheckPrimitiveMesh(std::make_shared<Box2>(Vector2{1.0f, 1.0f}), 0.25f, -0.25f);
    CheckPrimitiveMesh(std::make_shared<Sphere2>(0.75f), 0.50f, 0.0f);
    CheckPrimitiveMesh(std::make_shared<Capsule2>(0.75f, 0.5f), 0.0f, -0.5f);
    CheckPrimitiveMesh(std::make_shared<Convex2>(
        std::vector<Vector2>{{-0.75f, -0.75f}, {0.75f, -0.75f},
                             {0.75f, 0.75f}, {-0.75f, 0.75f}}), 0.50f, 0.0f);
}

R2_TEST("Triangle mesh to triangle mesh collision reports both child indices")
{
    const auto meshA = std::make_shared<TriangleMesh2>(
        std::vector<Vector2>{{-2.0f, -2.0f}, {1.0f, -2.0f}, {1.0f, 2.0f}},
        std::vector<std::uint32_t>{0, 1, 2});
    const auto meshB = std::make_shared<TriangleMesh2>(
        std::vector<Vector2>{{-1.0f, -2.0f}, {2.0f, -2.0f}, {-1.0f, 2.0f}},
        std::vector<std::uint32_t>{0, 1, 2});
    std::unique_ptr<RigidBody2D> bodyA = MakeStaticBody(meshA, Vector2::Zero());
    std::unique_ptr<RigidBody2D> bodyB = MakeStaticBody(meshB, {1.75f, 0.0f});
    Contact2 contactAB;
    Contact2 contactBA;
    R2_REQUIRE(GenerateContact(*bodyA, *bodyB, contactAB));
    R2_REQUIRE(GenerateContact(*bodyB, *bodyA, contactBA));
    R2_REQUIRE(contactAB.Normal.X > 0.9f && contactBA.Normal.X < -0.9f);
    R2_REQUIRE(contactAB.TriangleIndexA == 0 && contactAB.TriangleIndexB == 0);
    R2_REQUIRE(contactBA.TriangleIndexA == 0 && contactBA.TriangleIndexB == 0);
    bodyB->SetTransform({{2.25f, 0.0f}, 0.0f});
    R2_REQUIRE(!GenerateContact(*bodyA, *bodyB, contactAB));
}

R2_TEST("Convex2 and triangle mesh constructors reject invalid topology")
{
    bool convexRejected = false;
    try
    {
        Convex2 invalid({{0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}});
        (void)invalid;
    }
    catch (const std::invalid_argument&)
    {
        convexRejected = true;
    }
    R2_REQUIRE(convexRejected);

    bool meshRejected = false;
    try
    {
        TriangleMesh2 invalid({{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}}, {0, 1, 7});
        (void)invalid;
    }
    catch (const std::out_of_range&)
    {
        meshRejected = true;
    }
    R2_REQUIRE(meshRejected);
}
