#include "Capsule2.h"

#include "Maths/Maths.h"

#include <algorithm>

namespace Riemann2D
{
    Capsule2::Capsule2(float halfLength, float radius)
        : HalfLength(halfLength), Radius(radius)
    {
    }

    AxisAlignedBox2 Capsule2::ComputeAABB(const Transform2& transform) const
    {
        const Vector2 endpointA = transform.TransformPoint({-HalfLength, 0.0f});
        const Vector2 endpointB = transform.TransformPoint({HalfLength, 0.0f});
        const Vector2 extent{Radius, Radius};
        return {{std::min(endpointA.X, endpointB.X) - extent.X,
                 std::min(endpointA.Y, endpointB.Y) - extent.Y},
                {std::max(endpointA.X, endpointB.X) + extent.X,
                 std::max(endpointA.Y, endpointB.Y) + extent.Y}};
    }

    MassProperties2 Capsule2::ComputeMass(float density) const
    {
        const float rectangleMass = density * 4.0f * HalfLength * Radius;
        const float capsMass = density * Pi * Radius * Radius;
        const float rectangleInertia =
            rectangleMass * (HalfLength * HalfLength + Radius * Radius) / 3.0f;

        // Integrating the two outward-facing half-disks includes both their
        // centroid offsets and their inertia about their own centroids.
        const float capsInertia = density *
            (Pi * Radius * Radius * HalfLength * HalfLength +
             (8.0f / 3.0f) * HalfLength * Radius * Radius * Radius +
             0.5f * Pi * Radius * Radius * Radius * Radius);
        return {rectangleMass + capsMass, rectangleInertia + capsInertia, Vector2::Zero()};
    }

    Vector2 Capsule2::Support(const Vector2& localDirection) const
    {
        const Vector2 direction = localDirection.SafeUnit();
        if (direction.SquareLength() <= Epsilon)
        {
            return {HalfLength + Radius, 0.0f};
        }

        const float endpointX = localDirection.X >= 0.0f ? HalfLength : -HalfLength;
        return Vector2{endpointX, 0.0f} + direction * Radius;
    }
}
