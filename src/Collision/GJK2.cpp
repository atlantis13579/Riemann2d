#include "GJK2.h"

#include "Maths/Maths.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace Riemann2D
{
    namespace
    {
        constexpr int GJKMaxIterations = 32;
        constexpr int EPAMaxIterations = 64;
        constexpr float CollisionTolerance = 1.0e-5f;
        constexpr float CollisionToleranceSquared = CollisionTolerance * CollisionTolerance;

        struct SupportVertex
        {
            Vector2 Point = Vector2::Zero();
            Vector2 PointA = Vector2::Zero();
            Vector2 PointB = Vector2::Zero();
        };

        struct Simplex
        {
            std::array<SupportVertex, 3> Vertices{};
            int Count = 0;
        };

        struct EdgeCandidate
        {
            std::size_t Index = 0;
            Vector2 Normal = Vector2::UnitX();
            float Distance = std::numeric_limits<float>::max();
        };

        bool IsSupportedShape(const Shape2& shape)
        {
            return shape.GetType() != ShapeType2::TriangleMesh2;
        }

        bool IsFinite(const Vector2& value)
        {
            return std::isfinite(value.X) && std::isfinite(value.Y);
        }

        SupportVertex GetSupport(const Shape2& shapeA, const Transform2& transformA,
                                 const Shape2& shapeB, const Transform2& transformB,
                                 Vector2 direction)
        {
            if (direction.SquareLength() <= CollisionToleranceSquared || !IsFinite(direction))
            {
                direction = Vector2::UnitX();
            }

            SupportVertex vertex;
            vertex.PointA = transformA.TransformPoint(
                shapeA.Support(transformA.InverseTransformVector(direction)));
            vertex.PointB = transformB.TransformPoint(
                shapeB.Support(transformB.InverseTransformVector(-direction)));
            vertex.Point = vertex.PointA - vertex.PointB;
            return vertex;
        }

        bool SamePoint(const Vector2& a, const Vector2& b)
        {
            return (a - b).SquareLength() <= CollisionToleranceSquared;
        }

        Vector2 ClosestPointOnSegment(const Vector2& a, const Vector2& b, float& parameter)
        {
            const Vector2 edge = b - a;
            const float lengthSquared = edge.SquareLength();
            if (lengthSquared <= CollisionToleranceSquared)
            {
                parameter = 0.0f;
                return a;
            }
            parameter = Clamp(-a.Dot(edge) / lengthSquared, 0.0f, 1.0f);
            return a + parameter * edge;
        }

        bool ReduceLine(Simplex& simplex, Vector2& direction)
        {
            const SupportVertex a = simplex.Vertices[0];
            const SupportVertex b = simplex.Vertices[1];
            float parameter = 0.0f;
            const Vector2 closest = ClosestPointOnSegment(a.Point, b.Point, parameter);
            if (closest.SquareLength() <= CollisionToleranceSquared)
            {
                direction = Vector2::Zero();
                return true;
            }

            if (parameter <= CollisionTolerance)
            {
                simplex.Vertices[0] = a;
                simplex.Count = 1;
            }
            else if (parameter >= 1.0f - CollisionTolerance)
            {
                simplex.Vertices[0] = b;
                simplex.Count = 1;
            }
            direction = -closest;
            return false;
        }

        bool OriginInsideTriangle(const Vector2& a, const Vector2& b, const Vector2& c)
        {
            const float area = (b - a).Cross(c - a);
            if (std::abs(area) <= CollisionToleranceSquared)
            {
                return false;
            }

            const float ab = (b - a).Cross(-a);
            const float bc = (c - b).Cross(-b);
            const float ca = (a - c).Cross(-c);
            const float scale = std::max({(b - a).Length(), (c - b).Length(),
                                          (a - c).Length(), 1.0f});
            const float tolerance = CollisionTolerance * scale;
            if (area > 0.0f)
            {
                return ab >= -tolerance && bc >= -tolerance && ca >= -tolerance;
            }
            return ab <= tolerance && bc <= tolerance && ca <= tolerance;
        }

        bool ReduceTriangle(Simplex& simplex, Vector2& direction)
        {
            const std::array<SupportVertex, 3> vertices = simplex.Vertices;
            if (OriginInsideTriangle(vertices[0].Point, vertices[1].Point, vertices[2].Point))
            {
                direction = Vector2::Zero();
                return true;
            }

            float bestDistanceSquared = std::numeric_limits<float>::max();
            int bestFirst = 0;
            int bestSecond = 0;
            float bestParameter = 0.0f;
            for (int edgeIndex = 0; edgeIndex < 3; ++edgeIndex)
            {
                const int nextIndex = (edgeIndex + 1) % 3;
                float parameter = 0.0f;
                const Vector2 closest = ClosestPointOnSegment(
                    vertices[edgeIndex].Point, vertices[nextIndex].Point, parameter);
                const float distanceSquared = closest.SquareLength();
                if (distanceSquared < bestDistanceSquared)
                {
                    bestDistanceSquared = distanceSquared;
                    bestFirst = edgeIndex;
                    bestSecond = nextIndex;
                    bestParameter = parameter;
                }
            }

            if (bestDistanceSquared <= CollisionToleranceSquared)
            {
                direction = Vector2::Zero();
                return true;
            }

            if (bestParameter <= CollisionTolerance)
            {
                simplex.Vertices[0] = vertices[bestFirst];
                simplex.Count = 1;
                direction = -vertices[bestFirst].Point;
            }
            else if (bestParameter >= 1.0f - CollisionTolerance)
            {
                simplex.Vertices[0] = vertices[bestSecond];
                simplex.Count = 1;
                direction = -vertices[bestSecond].Point;
            }
            else
            {
                simplex.Vertices[0] = vertices[bestFirst];
                simplex.Vertices[1] = vertices[bestSecond];
                simplex.Count = 2;
                direction = -(vertices[bestFirst].Point +
                              bestParameter * (vertices[bestSecond].Point - vertices[bestFirst].Point));
            }
            return false;
        }

        bool ReduceSimplex(Simplex& simplex, Vector2& direction)
        {
            if (simplex.Count == 1)
            {
                direction = -simplex.Vertices[0].Point;
                return direction.SquareLength() <= CollisionToleranceSquared;
            }
            if (simplex.Count == 2)
            {
                return ReduceLine(simplex, direction);
            }
            return ReduceTriangle(simplex, direction);
        }

        bool RunGJK(const Shape2& shapeA, const Transform2& transformA,
                    const Shape2& shapeB, const Transform2& transformB,
                    Simplex& simplex)
        {
            simplex.Count = 0;
            if (!IsSupportedShape(shapeA) || !IsSupportedShape(shapeB))
            {
                return false;
            }

            Vector2 direction = transformB.Position - transformA.Position;
            if (direction.SquareLength() <= CollisionToleranceSquared)
            {
                direction = Vector2::UnitX();
            }

            simplex.Vertices[0] = GetSupport(shapeA, transformA, shapeB, transformB, direction);
            if (!IsFinite(simplex.Vertices[0].Point))
            {
                return false;
            }
            simplex.Count = 1;
            direction = -simplex.Vertices[0].Point;
            if (direction.SquareLength() <= CollisionToleranceSquared)
            {
                return true;
            }

            for (int iteration = 0; iteration < GJKMaxIterations; ++iteration)
            {
                const SupportVertex vertex = GetSupport(shapeA, transformA, shapeB, transformB, direction);
                if (!IsFinite(vertex.Point))
                {
                    return false;
                }

                const float scaledTolerance = CollisionTolerance * std::max(direction.Length(), 1.0f);
                if (vertex.Point.Dot(direction) < -scaledTolerance)
                {
                    return false;
                }

                bool duplicate = false;
                for (int index = 0; index < simplex.Count; ++index)
                {
                    duplicate = duplicate || SamePoint(vertex.Point, simplex.Vertices[index].Point);
                }
                if (duplicate)
                {
                    return direction.SquareLength() <= CollisionToleranceSquared;
                }

                simplex.Vertices[simplex.Count++] = vertex;
                if (ReduceSimplex(simplex, direction))
                {
                    return true;
                }
            }

            // The cap prevents pathological support maps from hanging the caller.
            // Only accept a capped search when its final separating distance is tiny.
            return direction.SquareLength() <= CollisionToleranceSquared;
        }

        float HullCross(const SupportVertex& origin, const SupportVertex& a, const SupportVertex& b)
        {
            return (a.Point - origin.Point).Cross(b.Point - origin.Point);
        }

        std::vector<SupportVertex> BuildHull(std::vector<SupportVertex> points)
        {
            std::sort(points.begin(), points.end(), [](const SupportVertex& a, const SupportVertex& b)
            {
                return a.Point.X < b.Point.X ||
                       (a.Point.X == b.Point.X && a.Point.Y < b.Point.Y);
            });
            points.erase(std::unique(points.begin(), points.end(), [](const SupportVertex& a,
                                                                      const SupportVertex& b)
            {
                return SamePoint(a.Point, b.Point);
            }), points.end());

            if (points.size() <= 2)
            {
                return points;
            }

            std::vector<SupportVertex> hull;
            hull.reserve(points.size() * 2);
            for (const SupportVertex& point : points)
            {
                while (hull.size() >= 2 &&
                       HullCross(hull[hull.size() - 2], hull.back(), point) <= CollisionToleranceSquared)
                {
                    hull.pop_back();
                }
                hull.push_back(point);
            }

            const std::size_t lowerSize = hull.size();
            for (std::size_t index = points.size() - 1; index-- > 0;)
            {
                const SupportVertex& point = points[index];
                while (hull.size() > lowerSize &&
                       HullCross(hull[hull.size() - 2], hull.back(), point) <= CollisionToleranceSquared)
                {
                    hull.pop_back();
                }
                hull.push_back(point);
            }
            hull.pop_back();
            return hull;
        }

        std::vector<SupportVertex> MakeInitialPolytope(
            const Shape2& shapeA, const Transform2& transformA,
            const Shape2& shapeB, const Transform2& transformB, const Simplex& simplex)
        {
            std::vector<SupportVertex> points;
            points.reserve(5);
            for (int index = 0; index < simplex.Count; ++index)
            {
                points.push_back(simplex.Vertices[index]);
            }

            if (simplex.Count == 2)
            {
                Vector2 perpendicular =
                    (simplex.Vertices[1].Point - simplex.Vertices[0].Point).PerpendicularCCW();
                if (perpendicular.SquareLength() <= CollisionToleranceSquared)
                {
                    perpendicular = Vector2::UnitY();
                }
                points.push_back(GetSupport(shapeA, transformA, shapeB, transformB, perpendicular));
                points.push_back(GetSupport(shapeA, transformA, shapeB, transformB, -perpendicular));
            }
            return BuildHull(std::move(points));
        }

        bool FindClosestEdge(const std::vector<SupportVertex>& polytope, EdgeCandidate& candidate)
        {
            candidate = {};
            candidate.Distance = std::numeric_limits<float>::max();
            bool found = false;
            for (std::size_t index = 0; index < polytope.size(); ++index)
            {
                const std::size_t next = (index + 1) % polytope.size();
                const Vector2 edge = polytope[next].Point - polytope[index].Point;
                if (edge.SquareLength() <= CollisionToleranceSquared)
                {
                    continue;
                }

                Vector2 normal{edge.Y, -edge.X};
                normal = normal.SafeUnit();
                float distance = normal.Dot(polytope[index].Point);
                if (distance < 0.0f && distance > -CollisionTolerance)
                {
                    distance = 0.0f;
                }
                if (distance >= 0.0f && distance < candidate.Distance)
                {
                    candidate.Index = index;
                    candidate.Normal = normal;
                    candidate.Distance = distance;
                    found = true;
                }
            }
            return found;
        }

        void BuildResultFromEdge(const std::vector<SupportVertex>& polytope,
                                 const EdgeCandidate& edge, ConvexCollisionResult2& result)
        {
            const SupportVertex& first = polytope[edge.Index];
            const SupportVertex& second = polytope[(edge.Index + 1) % polytope.size()];
            float parameter = 0.0f;
            ClosestPointOnSegment(first.Point, second.Point, parameter);
            const Vector2 pointA = first.PointA + parameter * (second.PointA - first.PointA);
            const Vector2 pointB = first.PointB + parameter * (second.PointB - first.PointB);
            result.Normal = edge.Normal;
            result.Point = 0.5f * (pointA + pointB);
            result.Penetration = std::max(edge.Distance, 0.0f);
        }

        bool RunEPA(const Shape2& shapeA, const Transform2& transformA,
                    const Shape2& shapeB, const Transform2& transformB,
                    const Simplex& simplex, ConvexCollisionResult2& result)
        {
            if (simplex.Count == 1)
            {
                result.Normal = (transformB.Position - transformA.Position).SafeUnit();
                if (result.Normal.SquareLength() <= CollisionToleranceSquared)
                {
                    result.Normal = Vector2::UnitX();
                }
                result.Point = 0.5f * (simplex.Vertices[0].PointA + simplex.Vertices[0].PointB);
                result.Penetration = 0.0f;
                return true;
            }

            std::vector<SupportVertex> polytope =
                MakeInitialPolytope(shapeA, transformA, shapeB, transformB, simplex);
            if (polytope.size() < 3)
            {
                const SupportVertex& first = simplex.Vertices[0];
                result.Normal = (transformB.Position - transformA.Position).SafeUnit();
                if (result.Normal.SquareLength() <= CollisionToleranceSquared)
                {
                    result.Normal = Vector2::UnitX();
                }
                result.Point = 0.5f * (first.PointA + first.PointB);
                result.Penetration = 0.0f;
                return true;
            }

            EdgeCandidate bestEdge;
            for (int iteration = 0; iteration < EPAMaxIterations; ++iteration)
            {
                if (!FindClosestEdge(polytope, bestEdge))
                {
                    return false;
                }

                const SupportVertex support =
                    GetSupport(shapeA, transformA, shapeB, transformB, bestEdge.Normal);
                if (!IsFinite(support.Point))
                {
                    return false;
                }
                const float projection = support.Point.Dot(bestEdge.Normal);
                const float gap = projection - bestEdge.Distance;

                const std::size_t next = (bestEdge.Index + 1) % polytope.size();
                if (gap <= CollisionTolerance ||
                    SamePoint(support.Point, polytope[bestEdge.Index].Point) ||
                    SamePoint(support.Point, polytope[next].Point))
                {
                    BuildResultFromEdge(polytope, bestEdge, result);
                    return true;
                }
                polytope.insert(polytope.begin() + static_cast<std::ptrdiff_t>(next), support);
            }

            // A useful conservative contact is preferable to failing after the cap.
            if (FindClosestEdge(polytope, bestEdge))
            {
                BuildResultFromEdge(polytope, bestEdge, result);
                return true;
            }
            return false;
        }
    }

    bool GJKOverlap2(const Shape2& shapeA, const Transform2& transformA,
                      const Shape2& shapeB, const Transform2& transformB)
    {
        Simplex simplex;
        return RunGJK(shapeA, transformA, shapeB, transformB, simplex);
    }

    bool CollideConvex2(const Shape2& shapeA, const Transform2& transformA,
                         const Shape2& shapeB, const Transform2& transformB,
                         ConvexCollisionResult2& result)
    {
        result = {};
        Simplex simplex;
        if (!RunGJK(shapeA, transformA, shapeB, transformB, simplex))
        {
            return false;
        }
        return RunEPA(shapeA, transformA, shapeB, transformB, simplex, result);
    }
}
