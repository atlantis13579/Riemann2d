#pragma once

#include "Maths/Vector2.h"

#include <vector>

namespace Riemann2D
{
    bool CutConvexPolygon2D(const std::vector<Vector2>& polygon,
                            const Vector2& linePoint, const Vector2& lineNormal,
                            std::vector<Vector2>& negativeSide,
                            std::vector<Vector2>& positiveSide);
    float PolygonArea2D(const std::vector<Vector2>& polygon);
    Vector2 PolygonCentroid2D(const std::vector<Vector2>& polygon);

    bool FindCompactDestructionSplit2D(const std::vector<Vector2>& polygon,
                                       int negativePieceCount, int positivePieceCount,
                                       int branch, std::vector<Vector2>& bestNegative,
                                       std::vector<Vector2>& bestPositive,
                                       float sizeVariation = 0.0f);
}
