#pragma once

#include "Shape2.h"

#include <vector>

namespace Riemann2D
{
    class Convex2 final : public Shape2
    {
    public:
        explicit Convex2(std::vector<Vector2> vertices);

        ShapeType2 GetType() const override { return ShapeType2::Convex2; }
        AxisAlignedBox2 ComputeAABB(const Transform2& transform) const override;
        MassProperties2 ComputeMass(float density) const override;
        Vector2 Support(const Vector2& localDirection) const override;

        const std::vector<Vector2>& GetVertices() const { return Vertices; }
        std::size_t GetVertexCount() const { return Vertices.size(); }

    private:
        std::vector<Vector2> Vertices;
    };
}
