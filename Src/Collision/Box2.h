#pragma once

#include "Shape2.h"

namespace Riemann2D
{
    class Box2 final : public Shape2
    {
    public:
        explicit Box2(const Vector2& halfExtent);

        ShapeType2 GetType() const override { return ShapeType2::Box; }
        AxisAlignedBox2 ComputeAABB(const Transform2& transform) const override;
        MassProperties2 ComputeMass(float density) const override;
        Vector2 Support(const Vector2& localDirection) const override;

        const Vector2& GetHalfExtent() const { return HalfExtent; }

    private:
        Vector2 HalfExtent;
    };
}
