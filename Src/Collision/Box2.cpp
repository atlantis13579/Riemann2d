#include "Box2.h"


#include <array>

namespace Riemann2D
{
    Box2::Box2(const Vector2& halfExtent)
        : HalfExtent(halfExtent)
    {
    }

    AxisAlignedBox2 Box2::ComputeAABB(const Transform2& transform) const
    {
        const std::array<Vector2, 4> corners = {
            Vector2{-HalfExtent.X, -HalfExtent.Y}, Vector2{HalfExtent.X, -HalfExtent.Y},
            Vector2{HalfExtent.X, HalfExtent.Y}, Vector2{-HalfExtent.X, HalfExtent.Y}
        };
        const Vector2 first = transform.TransformPoint(corners[0]);
        AxisAlignedBox2 bounds{first, first};
        for (std::size_t index = 1; index < corners.size(); ++index)
        {
            bounds.Expand(transform.TransformPoint(corners[index]));
        }
        return bounds;
    }

    MassProperties2 Box2::ComputeMass(float density) const
    {
        const float width = 2.0f * HalfExtent.X;
        const float height = 2.0f * HalfExtent.Y;
        const float mass = density * width * height;
        return {mass, mass * (width * width + height * height) / 12.0f, Vector2::Zero()};
    }

    Vector2 Box2::Support(const Vector2& localDirection) const
    {
        return {localDirection.X >= 0.0f ? HalfExtent.X : -HalfExtent.X,
                localDirection.Y >= 0.0f ? HalfExtent.Y : -HalfExtent.Y};
    }
}
