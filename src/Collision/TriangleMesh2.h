#pragma once

#include "Shape2.h"

#include <array>
#include <cstdint>
#include <vector>

namespace Riemann2D
{
    class TriangleMesh2 final : public Shape2
    {
    public:
        using Triangle = std::array<Vector2, 3>;

        TriangleMesh2(std::vector<Vector2> vertices, std::vector<std::uint32_t> indices);

        ShapeType2 GetType() const override { return ShapeType2::TriangleMesh2; }
        AxisAlignedBox2 ComputeAABB(const Transform2& transform) const override;
        MassProperties2 ComputeMass(float density) const override;
        Vector2 Support(const Vector2& localDirection) const override;

        const std::vector<Vector2>& GetVertices() const { return Vertices; }
        const std::vector<std::uint32_t>& GetIndices() const { return Indices; }
        std::size_t GetTriangleCount() const { return Indices.size() / 3; }
        Triangle GetTriangle(std::size_t triangleIndex) const;

    private:
        std::vector<Vector2> Vertices;
        std::vector<std::uint32_t> Indices;
    };
}
