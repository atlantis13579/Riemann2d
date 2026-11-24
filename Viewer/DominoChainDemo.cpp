#include "DominoChainDemo.h"

#include "Collision/Box2.h"
#include "Collision/Sphere2.h"

namespace Riemann2D::Viewer
{
    void BuildDominoChainDemo(PhysicsWorld2D& world)
    {
        const auto box = [&](Vector2 position, Vector2 halfExtent, BodyType2D type,
                             float angle = 0.0f,
                             float density = 1.0f, float friction = 0.45f)
        {
            RigidBodyDesc2D body;
            body.Type = type;
            body.Shape = std::make_shared<Box2>(halfExtent);
            body.Transform = {position, angle};
            body.Density = density;
            body.Friction = friction;
            body.Restitution = 0.0f;
            world.CreateBody(body);
        };
        const auto ball = [&](Vector2 position, float radius,
                              float density, float restitution)
        {
            RigidBodyDesc2D body;
            body.Shape = std::make_shared<Sphere2>(radius);
            body.Transform.Position = position;
            body.Density = density;
            body.Friction = 0.2f;
            body.Restitution = restitution;
            world.CreateBody(body);
        };

        constexpr auto fixed = BodyType2D::Static;
        constexpr auto dynamic = BodyType2D::Dynamic;
        box({10.0f, -0.3f}, {16.0f, 0.3f}, fixed);

        // Project the 3D chain onto XY; the falling cylinder becomes a heavy disk.
        box({3.0f, 0.0f}, {1.0f, 1.0f}, fixed, 40.0f * Pi / 180.0f);
        box({2.0f, 1.7f}, {5.0f, 0.25f}, dynamic);
        ball({-1.0f, 4.0f}, 0.5f, 1.0f, 0.7f);
        box({-2.0f, 4.0f}, {0.5f, 0.5f}, dynamic, 0.0f, 2.0f);
        ball({6.0f, 10.0f}, 0.5f, 100.0f / (Pi * 0.25f), 0.0f);
        // An inclined catcher redirects the launch toward the upper row in the XY plane.
        box({-1.0f, 11.8f}, {3.2f, 0.18f}, fixed, 54.0f * Pi / 180.0f);

        box({14.0f, 8.7f}, {6.5f, 0.2f}, fixed, 0.0f, 1.0f, 0.35f);
        for (int index = 0; index < 11; ++index)
            box({8.7f + index, 9.8f}, {0.16f, 0.9f}, dynamic);

        box({20.1f, 9.45f}, {0.5f, 0.55f}, dynamic, 0.0f, 0.65f, 0.12f);
        box({21.0f, 6.55f}, {3.367863f, 0.15f}, fixed,
            49.214179f * Pi / 180.0f, 1.0f, 0.1f);
        box({23.8f, 6.4f}, {0.18f, 2.7f}, fixed);

        box({14.2f, 3.1f}, {4.9f, 0.2f}, fixed, 0.0f, 1.0f, 0.35f);
        for (int index = 0; index < 10; ++index)
            box({18.2f - 0.85f * index, 4.1f}, {0.15f, 0.8f}, dynamic);

        ball({9.6f, 3.85f}, 0.55f, 1.0f, 0.1f);
        box({8.5f, 0.2f}, {1.2f, 0.2f}, fixed);
        box({7.25f, 0.75f}, {0.12f, 0.75f}, fixed);
        box({9.75f, 0.75f}, {0.12f, 0.75f}, fixed);
    }
}
