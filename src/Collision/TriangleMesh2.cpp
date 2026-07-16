#include "TriangleMesh2.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace Riemann2D
{
    namespace
    {
        double TwiceTriangleArea(const Vector2& a, const Vector2& b, const Vector2& c)
        {
            return (static_cast<double>(b.X) - a.X) * (static_cast<double>(c.Y) - a.Y) -
                   (static_cast<double>(b.Y) - a.Y) * (static_cast<double>(c.X) - a.X);
        }

        void ValidateMesh(const std::vector<Vector2>& vertices,
                          const std::vector<std::uint32_t>& indices)
        {
            if (vertices.size() < 3 || indices.empty() || indices.size() % 3 != 0)
            {
                throw std::invalid_argument(
                    "A triangle mesh requires vertices and complete triangle indices");
            }

            double minimumX = vertices.front().X;
            double minimumY = vertices.front().Y;
            double maximumX = vertices.front().X;
            double maximumY = vertices.front().Y;
            for (const Vector2& vertex : vertices)
            {
                if (!std::isfinite(vertex.X) || !std::isfinite(vertex.Y))
                {
                    throw std::invalid_argument("Triangle mesh vertices must be finite");
                }
                minimumX = std::min(minimumX, static_cast<double>(vertex.X));
                minimumY = std::min(minimumY, static_cast<double>(vertex.Y));
                maximumX = std::max(maximumX, static_cast<double>(vertex.X));
                maximumY = std::max(maximumY, static_cast<double>(vertex.Y));
            }

            const double coordinateScale = std::max(
                1.0e-12, std::max(maximumX - minimumX, maximumY - minimumY));
            const double areaTolerance = 1.0e-7 * coordinateScale * coordinateScale;
            for (std::size_t index = 0; index < indices.size(); index += 3)
            {
                const std::uint32_t ia = indices[index];
                const std::uint32_t ib = indices[index + 1];
                const std::uint32_t ic = indices[index + 2];
                if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size())
                {
                    throw std::out_of_range("Triangle mesh index is outside the vertex array");
                }
                if (std::abs(TwiceTriangleArea(vertices[ia], vertices[ib], vertices[ic])) <=
                    areaTolerance)
                {
                    throw std::invalid_argument("Triangle mesh contains a degenerate triangle");
                }
            }
        }
    }

    TriangleMesh2::TriangleMesh2(std::vector<Vector2> vertices,
                                   std::vector<std::uint32_t> indices)
        : Vertices(std::move(vertices)), Indices(std::move(indices))
    {
        ValidateMesh(Vertices, Indices);
    }

    AxisAlignedBox2 TriangleMesh2::ComputeAABB(const Transform2& transform) const
    {
        const Vector2 first = transform.TransformPoint(Vertices.front());
        AxisAlignedBox2 bounds{first, first};
        for (std::size_t index = 1; index < Vertices.size(); ++index)
        {
            bounds.Expand(transform.TransformPoint(Vertices[index]));
        }
        return bounds;
    }

    MassProperties2 TriangleMesh2::ComputeMass(float density) const
    {
        const double referenceX = Vertices.front().X;
        const double referenceY = Vertices.front().Y;
        double totalArea = 0.0;
        double weightedCenterX = 0.0;
        double weightedCenterY = 0.0;
        double inertiaAtOrigin = 0.0;

        for (std::size_t triangleIndex = 0; triangleIndex < GetTriangleCount(); ++triangleIndex)
        {
            const Triangle triangle = GetTriangle(triangleIndex);
            const Vector2& a = triangle[0];
            const Vector2& b = triangle[1];
            const Vector2& c = triangle[2];
            const double area = 0.5 * std::abs(TwiceTriangleArea(a, b, c));
            const double ax = static_cast<double>(a.X) - referenceX;
            const double ay = static_cast<double>(a.Y) - referenceY;
            const double bx = static_cast<double>(b.X) - referenceX;
            const double by = static_cast<double>(b.Y) - referenceY;
            const double cx = static_cast<double>(c.X) - referenceX;
            const double cy = static_cast<double>(c.Y) - referenceY;
            const double centerX = (ax + bx + cx) / 3.0;
            const double centerY = (ay + by + cy) / 3.0;
            const double quadraticSum =
                ax * ax + bx * bx + cx * cx + ax * bx + bx * cx + cx * ax +
                ay * ay + by * by + cy * cy + ay * by + by * cy + cy * ay;

            totalArea += area;
            weightedCenterX += area * centerX;
            weightedCenterY += area * centerY;
            inertiaAtOrigin += static_cast<double>(density) * area * quadraticSum / 6.0;
        }

        const double centerX = weightedCenterX / totalArea;
        const double centerY = weightedCenterY / totalArea;
        const double mass = static_cast<double>(density) * totalArea;
        const double inertiaAtCenter = inertiaAtOrigin - mass *
            (centerX * centerX + centerY * centerY);
        return {static_cast<float>(mass),
                static_cast<float>(std::max(0.0, inertiaAtCenter)),
                {static_cast<float>(referenceX + centerX),
                 static_cast<float>(referenceY + centerY)}};
    }

    Vector2 TriangleMesh2::Support(const Vector2& localDirection) const
    {
        const Vector2* bestVertex = &Vertices.front();
        float bestProjection = bestVertex->Dot(localDirection);
        for (std::size_t index = 1; index < Vertices.size(); ++index)
        {
            const float projection = Vertices[index].Dot(localDirection);
            if (projection > bestProjection)
            {
                bestProjection = projection;
                bestVertex = &Vertices[index];
            }
        }
        return *bestVertex;
    }

    TriangleMesh2::Triangle TriangleMesh2::GetTriangle(std::size_t triangleIndex) const
    {
        if (triangleIndex >= GetTriangleCount())
        {
            throw std::out_of_range("Triangle mesh triangle index is out of range");
        }

        const std::size_t firstIndex = triangleIndex * 3;
        return {Vertices[Indices[firstIndex]], Vertices[Indices[firstIndex + 1]],
                Vertices[Indices[firstIndex + 2]]};
    }
}
