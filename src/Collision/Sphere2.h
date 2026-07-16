#pragma once

#include "Shape2.h"

namespace Riemann2D
{
    class Sphere2 final : public Shape2
    {
    public:
        explicit Sphere2(float radius);

        ShapeType2 GetType() const override { return ShapeType2::Sphere2; }
        AxisAlignedBox2 ComputeAABB(const Transform2& transform) const override;
        MassProperties2 ComputeMass(float density) const override;
        Vector2 Support(const Vector2& localDirection) const override;

        float GetRadius() const { return Radius; }

    private:
        float Radius;
    };
}
