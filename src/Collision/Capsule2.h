#pragma once

#include "Shape2.h"

namespace Riemann2D
{
    // A capsule whose medial segment lies on the local X axis.
    class Capsule2 final : public Shape2
    {
    public:
        Capsule2(float halfLength, float radius);

        ShapeType2 GetType() const override { return ShapeType2::Capsule2; }
        AxisAlignedBox2 ComputeAABB(const Transform2& transform) const override;
        MassProperties2 ComputeMass(float density) const override;
        Vector2 Support(const Vector2& localDirection) const override;

        float GetHalfLength() const { return HalfLength; }
        float GetRadius() const { return Radius; }

    private:
        float HalfLength;
        float Radius;
    };
}
