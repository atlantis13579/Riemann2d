#pragma once

#include "Collision/Shape2.h"

namespace Riemann2D
{
    struct ConvexCollisionResult2
    {
        Vector2 Normal = Vector2::UnitX();
        Vector2 Point = Vector2::Zero();
        float Penetration = 0.0f;
    };

    // Tests two support-mapped convex shapes. TriangleMesh2 shapes must be handled by
    // testing the convex shape against individual mesh triangles.
    bool GJKOverlap2(const Shape2& shapeA, const Transform2& transformA,
                      const Shape2& shapeB, const Transform2& transformB);

    // Computes one representative contact using GJK followed by EPA. The
    // returned normal points from A to B and penetration is non-negative.
    bool CollideConvex2(const Shape2& shapeA, const Transform2& transformA,
                         const Shape2& shapeB, const Transform2& transformB,
                         ConvexCollisionResult2& result);
}
