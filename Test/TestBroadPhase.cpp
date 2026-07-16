#include "Test.h"

#include "Collision/Box2.h"
#include "Collision/SAT2.h"
#include "Collision/NarrowPhase2.h"
#include "Dynamics/RigidBody2D.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

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
