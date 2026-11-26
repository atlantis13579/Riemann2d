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
#include "Collision/SAT2.h"
#include "Collision/NarrowPhase2.h"
#include <algorithm>
#include <cstdint>
#include "Collision/StaticAABBTree2.h"
#include "Collision/DynamicAABBTree2.h"

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

namespace TestBroadPhaseCases
{
using namespace Riemann2D;

R2_TEST("Sweep and prune pairs match brute force AABB overlap")
{
    std::vector<BroadPhaseProxy2> proxies;
    for (std::uint32_t index = 0; index < 96; ++index)
    {
        const float x = static_cast<float>((index * 23) % 41) * 0.45f;
        const float y = static_cast<float>((index * 31) % 37) * 0.4f;
        proxies.push_back({index, {{x, y}, {x + 1.1f, y + 0.9f}}, index % 7 == 0});
    }

    std::vector<BroadPhasePair2> expected;
    for (std::size_t a = 0; a < proxies.size(); ++a)
    {
        for (std::size_t b = a + 1; b < proxies.size(); ++b)
        {
            if (!(proxies[a].IsStatic && proxies[b].IsStatic) &&
                proxies[a].Bounds.Overlaps(proxies[b].Bounds))
            {
                expected.push_back({proxies[a].Id, proxies[b].Id});
            }
        }
    }
    std::sort(expected.begin(), expected.end(), [](const BroadPhasePair2& left,
                                                   const BroadPhasePair2& right)
    {
        return left.A < right.A || (left.A == right.A && left.B < right.B);
    });

    SweepAndPruneBroadPhase2 broadPhase;
    std::vector<BroadPhasePair2> actual;
    broadPhase.ComputePairs(proxies, actual);
    R2_REQUIRE(actual == expected);
}

R2_TEST("Narrow phase filters broad phase false positives")
{
    const auto shape = std::make_shared<Box2>(Vector2{1.0f, 0.2f});
    std::vector<std::unique_ptr<RigidBody2D>> bodies;
    RigidBodyDesc2D description;
    description.Type = BodyType2D::Static;
    description.Shape = shape;
    description.Transform.Angle = 0.785398163f;
    bodies.push_back(std::make_unique<RigidBody2D>(description));
    description.Transform.Position = {0.0f, 1.5f};
    bodies.push_back(std::make_unique<RigidBody2D>(description));

    const AxisAlignedBox2 boundsA = shape->ComputeAABB(bodies[0]->GetTransform());
    const AxisAlignedBox2 boundsB = shape->ComputeAABB(bodies[1]->GetTransform());
    R2_REQUIRE(boundsA.Overlaps(boundsB));

    NarrowPhase2 narrowPhase;
    std::vector<Contact2> contacts;
    narrowPhase.GenerateContacts(bodies, {{0, 1}}, contacts);
    R2_REQUIRE(contacts.empty());
}
}

namespace TestTreesCases
{
using namespace Riemann2D;

R2_TEST("Static tree queries match brute force results")
{
    std::vector<StaticAABBTree2::Item> items;
    for (std::uint32_t y = 0; y < 10; ++y)
    {
        for (std::uint32_t x = 0; x < 10; ++x)
        {
            const Vector2 minimum{static_cast<float>(x) * 1.5f, static_cast<float>(y) * 1.25f};
            items.push_back({{minimum, minimum + Vector2{0.8f, 0.7f}}, y * 10 + x});
        }
    }

    StaticAABBTree2 tree;
    R2_REQUIRE(tree.Build(items));
    R2_REQUIRE(!tree.Empty());
    R2_REQUIRE(tree.GetNodeCount() == items.size() * 2 - 1);
    R2_REQUIRE(tree.GetHeight() <= 8);
    R2_REQUIRE(tree.Validate());

    const std::vector<AxisAlignedBox2> queries = {
        {{-1.0f, -1.0f}, {2.0f, 2.0f}},
        {{4.2f, 3.7f}, {10.1f, 8.4f}},
        {{13.0f, 10.0f}, {20.0f, 20.0f}},
        {{100.0f, 100.0f}, {101.0f, 101.0f}}
    };
    for (const AxisAlignedBox2& query : queries)
    {
        std::vector<std::uint32_t> expected;
        for (const StaticAABBTree2::Item& item : items)
        {
            if (item.Bounds.Overlaps(query)) expected.push_back(item.Payload);
        }
        std::vector<std::uint32_t> actual;
        tree.Query(query, actual);
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        R2_REQUIRE(actual == expected);
    }
}

R2_TEST("Static tree rejects invalid bounds and clears prior state")
{
    StaticAABBTree2 tree;
    R2_REQUIRE(tree.Build({{{{0.0f, 0.0f}, {1.0f, 1.0f}}, 7}}));
    R2_REQUIRE(!tree.Build({{{{2.0f, 0.0f}, {1.0f, 1.0f}}, 9}}));
    R2_REQUIRE(tree.Empty());
    R2_REQUIRE(tree.Validate());
}

R2_TEST("Dynamic tree inserts moves queries and removes proxies")
{
    DynamicAABBTree2 tree;
    const ProxyHandle first = tree.CreateProxy({{0.0f, 0.0f}, {1.0f, 1.0f}});
    const ProxyHandle second = tree.CreateProxy({{2.0f, 0.0f}, {3.0f, 1.0f}});
    const ProxyHandle third = tree.CreateProxy({{5.0f, 0.0f}, {6.0f, 1.0f}});
    R2_REQUIRE(first.IsValid() && second.IsValid() && third.IsValid());
    R2_REQUIRE(tree.GetProxyCount() == 3);
    R2_REQUIRE(tree.Validate());

    std::vector<ProxyHandle> hits = tree.Query({{-0.5f, -0.5f}, {3.2f, 1.5f}});
    R2_REQUIRE(hits.size() == 2);
    R2_REQUIRE(std::find(hits.begin(), hits.end(), first) != hits.end());
    R2_REQUIRE(std::find(hits.begin(), hits.end(), second) != hits.end());

    R2_REQUIRE(!tree.MoveProxy(first, {{0.02f, 0.0f}, {1.02f, 1.0f}}, {0.02f, 0.0f}));
    R2_REQUIRE(tree.MoveProxy(third, {{1.2f, 2.0f}, {2.2f, 3.0f}}, {-3.8f, 2.0f}));
    hits = tree.Query({{1.0f, 1.8f}, {2.4f, 3.2f}});
    R2_REQUIRE(std::find(hits.begin(), hits.end(), third) != hits.end());
    R2_REQUIRE(tree.Validate());

    R2_REQUIRE(tree.DestroyProxy(second));
    R2_REQUIRE(!tree.IsValidProxy(second));
    R2_REQUIRE(!tree.DestroyProxy(second));
    R2_REQUIRE(tree.GetProxyCount() == 2);
    R2_REQUIRE(tree.Validate());
}

R2_TEST("Dynamic tree generations invalidate recycled handles")
{
    DynamicAABBTree2 tree;
    const ProxyHandle oldHandle = tree.CreateProxy({{0.0f, 0.0f}, {1.0f, 1.0f}});
    R2_REQUIRE(tree.DestroyProxy(oldHandle));
    const ProxyHandle newHandle = tree.CreateProxy({{2.0f, 2.0f}, {3.0f, 3.0f}});
    R2_REQUIRE(newHandle.IsValid());
    R2_REQUIRE(newHandle != oldHandle);
    R2_REQUIRE(!tree.IsValidProxy(oldHandle));
    R2_REQUIRE(tree.IsValidProxy(newHandle));
}

R2_TEST("Dynamic tree randomized queries never miss tight bounds")
{
    DynamicAABBTree2 tree;
    std::vector<ProxyHandle> handles;
    std::vector<AxisAlignedBox2> bounds;
    for (int index = 0; index < 128; ++index)
    {
        const float x = static_cast<float>((index * 37) % 53) * 0.7f;
        const float y = static_cast<float>((index * 19) % 47) * 0.6f;
        const Vector2 extent{0.2f + 0.03f * static_cast<float>(index % 7),
                             0.25f + 0.02f * static_cast<float>(index % 5)};
        bounds.push_back({{x, y}, {x + extent.X, y + extent.Y}});
        handles.push_back(tree.CreateProxy(bounds.back()));
    }
    R2_REQUIRE(tree.Validate());

    for (int queryIndex = 0; queryIndex < 40; ++queryIndex)
    {
        const float x = static_cast<float>((queryIndex * 11) % 31) * 0.9f;
        const float y = static_cast<float>((queryIndex * 7) % 29) * 0.8f;
        const AxisAlignedBox2 query{{x, y}, {x + 4.0f, y + 3.0f}};
        const std::vector<ProxyHandle> hits = tree.Query(query);
        for (std::size_t index = 0; index < bounds.size(); ++index)
        {
            if (bounds[index].Overlaps(query))
            {
                R2_REQUIRE(std::find(hits.begin(), hits.end(), handles[index]) != hits.end());
            }
        }
    }

    for (std::size_t index = 0; index < handles.size(); index += 3)
    {
        const Vector2 displacement{5.0f, -2.0f};
        bounds[index].Minimum += displacement;
        bounds[index].Maximum += displacement;
        R2_REQUIRE(tree.MoveProxy(handles[index], bounds[index], displacement));
    }
    for (std::size_t index = 1; index < handles.size(); index += 4)
    {
        R2_REQUIRE(tree.DestroyProxy(handles[index]));
    }
    R2_REQUIRE(tree.Validate());
}
}
