#include "Test.h"

#include "Collision/Box2.h"
#include "Destruction/DestructionCut2D.h"
#include "Destruction/DestructionSet2D.h"
#include "Dynamics/PhysicsWorld2D.h"

#include <cmath>
#include <memory>

using namespace Riemann2D;

namespace
{
    float PolygonPerimeter(const std::vector<Vector2>& polygon)
    {
        float perimeter = 0.0f;
        for (std::size_t index = 0; index < polygon.size(); ++index)
            perimeter += (polygon[(index + 1) % polygon.size()] - polygon[index]).Length();
        return perimeter;
    }
}

R2_TEST("Four-piece polygon fracture preserves area and avoids thin strips")
{
    DestructionSet2D destruction;
    destruction.CreateBox({0.0f, 2.0f}, {2.0f, 3.0f}, 4);
    const float intactArea = PolygonArea2D(destruction.GetIntactVertices());
    R2_REQUIRE(destruction.Fracture({0.0f, 5.0f}, 3.0f));
    R2_REQUIRE(destruction.GetPieces().size() == 4);
    float fragmentArea = 0.0f;
    for (const DestructionPiece2D& piece : destruction.GetPieces())
    {
        R2_REQUIRE(piece.LocalVertices.size() >= 3);
        const float area = PolygonArea2D(piece.LocalVertices);
        const float perimeter = PolygonPerimeter(piece.LocalVertices);
        R2_REQUIRE(area > 0.45f * intactArea / 4.0f);
        R2_REQUIRE(perimeter * perimeter / area < 32.0f);
        fragmentArea += area;
    }
    R2_REQUIRE(NearlyEqual(intactArea, fragmentArea, 1.0e-3f));
}

R2_TEST("Physics-backed destruction fragments keep blocking projectiles")
{
    PhysicsWorldSettings2D settings;
    settings.Gravity = Vector2::Zero();
    PhysicsWorld2D world(settings);
    RigidBodyDesc2D projectileDescription;
    projectileDescription.Shape = std::make_shared<Box2>(Vector2{0.6f, 0.6f});
    projectileDescription.Transform.Position = {0.0f, 4.0f};
    projectileDescription.LinearVelocity = {0.0f, -8.0f};
    RigidBody2D& projectile = world.CreateBody(projectileDescription);

    DestructionSet2D destruction;
    destruction.CreateBox(Vector2::Zero(), {2.0f, 1.5f}, 4);
    destruction.SetPhysicsWorld(&world);
    R2_REQUIRE(destruction.Fracture({0.0f, 3.0f}, 0.0f));
    for (const DestructionPiece2D& piece : destruction.GetPieces())
        R2_REQUIRE(piece.Body != nullptr);

    bool blocked = false;
    for (int step = 0; step < 120; ++step)
    {
        world.Step(1.0f / 120.0f);
        for (const Contact2& contact : world.GetContacts())
        {
            if (contact.BodyA == &projectile || contact.BodyB == &projectile)
            {
                blocked = true;
                break;
            }
        }
        if (blocked) break;
    }
    R2_REQUIRE(blocked);
}

R2_TEST("Destruction fragments simulate gravity and ground collision")
{
    DestructionSet2D destruction;
    destruction.CreateBox({0.0f, 3.0f}, {1.0f, 1.0f}, 2);
    destruction.Fracture({0.0f, 4.0f}, 1.0f);
    for (int step = 0; step < 240; ++step) destruction.Step(1.0f / 120.0f);
    for (const DestructionPiece2D& piece : destruction.GetPieces())
    {
        R2_REQUIRE(std::isfinite(piece.Position.X) && std::isfinite(piece.Position.Y));
        for (const Vector2& vertex : piece.LocalVertices)
            R2_REQUIRE(piece.Position.Y + vertex.Rotated(piece.Angle).Y >=
                       destruction.GetSettings().GroundHeight - 1.0e-3f);
    }
}
