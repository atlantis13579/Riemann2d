#pragma once

#include "Collision/AxisAlignedBox2.h"
#include "Maths/Transform2.h"

#include <cstdint>

namespace Riemann2D
{
    enum class ShapeType2 : std::uint8_t
    {
        Box,
        Sphere2,
        Capsule2,
        Convex2,
        TriangleMesh2
    };

    struct MassProperties2
    {
        float Mass = 0.0f;
        float Inertia = 0.0f;
        Vector2 Center = Vector2::Zero();
    };

    class Shape2
    {
    public:
        virtual ~Shape2() = default;

        virtual ShapeType2 GetType() const = 0;
        virtual AxisAlignedBox2 ComputeAABB(const Transform2& transform) const = 0;
        virtual MassProperties2 ComputeMass(float density) const = 0;
        virtual Vector2 Support(const Vector2& localDirection) const = 0;
    };
}
