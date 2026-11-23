#include "Sphere2.h"

#include "Maths/Maths.h"

namespace Riemann2D
{
    Sphere2::Sphere2(float radius)
        : Radius(radius)
    {
    }

    AxisAlignedBox2 Sphere2::ComputeAABB(const Transform2& transform) const
    {
        const Vector2 extent{Radius, Radius};
        return {transform.Position - extent, transform.Position + extent};
    }

    MassProperties2 Sphere2::ComputeMass(float density) const
    {
        const float mass = density * Pi * Radius * Radius;
        return {mass, 0.5f * mass * Radius * Radius, Vector2::Zero()};
    }

    Vector2 Sphere2::Support(const Vector2& localDirection) const
    {
        const Vector2 direction = localDirection.SafeUnit();
        return direction.SquareLength() > Epsilon ? direction * Radius : Vector2{Radius, 0.0f};
    }
}
