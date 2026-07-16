#include "Convex2.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace Riemann2D
{
    namespace
    {
        double Cross(const Vector2& a, const Vector2& b, const Vector2& c)
        {
            return (static_cast<double>(b.X) - a.X) * (static_cast<double>(c.Y) - a.Y) -
                   (static_cast<double>(b.Y) - a.Y) * (static_cast<double>(c.X) - a.X);
        }

        double SignedTwiceArea(const std::vector<Vector2>& vertices)
        {
            const double referenceX = vertices.front().X;
            const double referenceY = vertices.front().Y;
            double area = 0.0;
            for (std::size_t index = 0; index < vertices.size(); ++index)
            {
                const Vector2& current = vertices[index];
                const Vector2& next = vertices[(index + 1) % vertices.size()];
                const double currentX = static_cast<double>(current.X) - referenceX;
                const double currentY = static_cast<double>(current.Y) - referenceY;
                const double nextX = static_cast<double>(next.X) - referenceX;
                const double nextY = static_cast<double>(next.Y) - referenceY;
                area += currentX * nextY - currentY * nextX;
            }
            return area;
        }

        void ValidateAndOrient(std::vector<Vector2>& vertices)
        {
            if (vertices.size() < 3)
            {
                throw std::invalid_argument("A convex shape requires at least three vertices");
            }

            double minimumX = vertices.front().X;
            double minimumY = vertices.front().Y;
            double maximumX = vertices.front().X;
            double maximumY = vertices.front().Y;
            for (const Vector2& vertex : vertices)
            {
                if (!std::isfinite(vertex.X) || !std::isfinite(vertex.Y))
                {
                    throw std::invalid_argument("Convex2 shape vertices must be finite");
                }
                minimumX = std::min(minimumX, static_cast<double>(vertex.X));
                minimumY = std::min(minimumY, static_cast<double>(vertex.Y));
                maximumX = std::max(maximumX, static_cast<double>(vertex.X));
                maximumY = std::max(maximumY, static_cast<double>(vertex.Y));
            }

            const double coordinateScale = std::max(
                1.0e-12, std::max(maximumX - minimumX, maximumY - minimumY));
            const double areaTolerance = 1.0e-7 * coordinateScale * coordinateScale;
            double twiceArea = SignedTwiceArea(vertices);
            if (std::abs(twiceArea) <= areaTolerance)
            {
                throw std::invalid_argument("Convex2 shape vertices form a degenerate polygon");
            }
            if (twiceArea < 0.0)
            {
                std::reverse(vertices.begin(), vertices.end());
            }

            for (std::size_t index = 0; index < vertices.size(); ++index)
            {
                const Vector2& a = vertices[index];
                const Vector2& b = vertices[(index + 1) % vertices.size()];
                const Vector2& c = vertices[(index + 2) % vertices.size()];
                if (Cross(a, b, c) <= areaTolerance)
                {
                    throw std::invalid_argument(
                        "Convex2 shape vertices must describe a strictly convex polygon");
                }
            }
        }
    }

    Convex2::Convex2(std::vector<Vector2> vertices)
        : Vertices(std::move(vertices))
    {
        ValidateAndOrient(Vertices);
    }

    AxisAlignedBox2 Convex2::ComputeAABB(const Transform2& transform) const
    {
        const Vector2 first = transform.TransformPoint(Vertices.front());
        AxisAlignedBox2 bounds{first, first};
        for (std::size_t index = 1; index < Vertices.size(); ++index)
        {
            bounds.Expand(transform.TransformPoint(Vertices[index]));
        }
        return bounds;
    }

    MassProperties2 Convex2::ComputeMass(float density) const
    {
        const double referenceX = Vertices.front().X;
        const double referenceY = Vertices.front().Y;
        double twiceArea = 0.0;
        double centroidXNumerator = 0.0;
        double centroidYNumerator = 0.0;
        double inertiaIntegral = 0.0;

        for (std::size_t index = 0; index < Vertices.size(); ++index)
        {
            const Vector2& a = Vertices[index];
            const Vector2& b = Vertices[(index + 1) % Vertices.size()];
            const double ax = static_cast<double>(a.X) - referenceX;
            const double ay = static_cast<double>(a.Y) - referenceY;
            const double bx = static_cast<double>(b.X) - referenceX;
            const double by = static_cast<double>(b.Y) - referenceY;
            const double edgeCross = ax * by - ay * bx;
            twiceArea += edgeCross;
            centroidXNumerator += (ax + bx) * edgeCross;
            centroidYNumerator += (ay + by) * edgeCross;
            inertiaIntegral += edgeCross *
                (ax * ax + ax * bx + bx * bx + ay * ay + ay * by + by * by);
        }

        const double area = 0.5 * twiceArea;
        const double mass = static_cast<double>(density) * area;
        const double localCenterX = centroidXNumerator / (3.0 * twiceArea);
        const double localCenterY = centroidYNumerator / (3.0 * twiceArea);
        const double inertiaAtOrigin = static_cast<double>(density) * inertiaIntegral / 12.0;
        const double inertiaAtCenter = inertiaAtOrigin - mass *
            (localCenterX * localCenterX + localCenterY * localCenterY);

        return {static_cast<float>(mass),
                static_cast<float>(std::max(0.0, inertiaAtCenter)),
                {static_cast<float>(referenceX + localCenterX),
                 static_cast<float>(referenceY + localCenterY)}};
    }

    Vector2 Convex2::Support(const Vector2& localDirection) const
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
}
