#include "DestructionCut2D.h"

#include "Maths/Maths.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace Riemann2D
{
    namespace
    {
        std::vector<Vector2> ClipHalfPlane(const std::vector<Vector2>& polygon,
                                           const Vector2& point, const Vector2& normal,
                                           bool keepPositive)
        {
            std::vector<Vector2> output;
            if (polygon.empty()) return output;
            for (std::size_t index = 0; index < polygon.size(); ++index)
            {
                const Vector2 current = polygon[index];
                const Vector2 next = polygon[(index + 1) % polygon.size()];
                const float currentDistance = (current - point).Dot(normal);
                const float nextDistance = (next - point).Dot(normal);
                const bool currentInside = keepPositive ? currentDistance >= -Epsilon :
                                                        currentDistance <= Epsilon;
                const bool nextInside = keepPositive ? nextDistance >= -Epsilon :
                                                     nextDistance <= Epsilon;
                if (currentInside) output.push_back(current);
                if (currentInside != nextInside)
                {
                    const float denominator = currentDistance - nextDistance;
                    if (std::abs(denominator) > Epsilon)
                    {
                        const float fraction = currentDistance / denominator;
                        output.push_back(current + fraction * (next - current));
                    }
                }
            }
            return output;
        }

        float PolygonPerimeter(const std::vector<Vector2>& polygon)
        {
            float perimeter = 0.0f;
            for (std::size_t index = 0; index < polygon.size(); ++index)
                perimeter += (polygon[(index + 1) % polygon.size()] - polygon[index]).Length();
            return perimeter;
        }

        float CompactnessPenalty(const std::vector<Vector2>& polygon)
        {
            const float area = PolygonArea2D(polygon);
            const float perimeter = PolygonPerimeter(polygon);
            return area > Epsilon ? perimeter * perimeter / area : FLT_MAX;
        }

        float ProjectionRadius(const std::vector<Vector2>& polygon,
                               const Vector2& center, const Vector2& axis)
        {
            float radius = 0.0f;
            for (const Vector2& vertex : polygon)
                radius = std::max(radius, std::abs((vertex - center).Dot(axis)));
            return radius;
        }

        float EdgeAlignmentPenalty(const std::vector<Vector2>& polygon,
                                   const Vector2& cutNormal)
        {
            const Vector2 cutDirection = cutNormal.PerpendicularCCW();
            float maximumAlignment = 0.0f;
            for (std::size_t index = 0; index < polygon.size(); ++index)
            {
                const Vector2 edge =
                    (polygon[(index + 1) % polygon.size()] - polygon[index]).SafeUnit();
                maximumAlignment = std::max(maximumAlignment,
                                            std::abs(edge.Dot(cutDirection)));
            }
            return maximumAlignment * maximumAlignment;
        }
    }

    bool CutConvexPolygon2D(const std::vector<Vector2>& polygon,
                            const Vector2& linePoint, const Vector2& lineNormal,
                            std::vector<Vector2>& negativeSide,
                            std::vector<Vector2>& positiveSide)
    {
        const Vector2 normal = lineNormal.SafeUnit();
        if (polygon.size() < 3 || normal.SquareLength() <= Epsilon)
        {
            negativeSide.clear();
            positiveSide.clear();
            return false;
        }
        negativeSide = ClipHalfPlane(polygon, linePoint, normal, false);
        positiveSide = ClipHalfPlane(polygon, linePoint, normal, true);
        return negativeSide.size() >= 3 && positiveSide.size() >= 3 &&
               PolygonArea2D(negativeSide) > Epsilon && PolygonArea2D(positiveSide) > Epsilon;
    }

    float PolygonArea2D(const std::vector<Vector2>& polygon)
    {
        float twiceArea = 0.0f;
        for (std::size_t index = 0; index < polygon.size(); ++index)
            twiceArea += polygon[index].Cross(polygon[(index + 1) % polygon.size()]);
        return 0.5f * std::abs(twiceArea);
    }

    Vector2 PolygonCentroid2D(const std::vector<Vector2>& polygon)
    {
        float crossSum = 0.0f;
        Vector2 weighted = Vector2::Zero();
        for (std::size_t index = 0; index < polygon.size(); ++index)
        {
            const Vector2& current = polygon[index];
            const Vector2& next = polygon[(index + 1) % polygon.size()];
            const float cross = current.Cross(next);
            crossSum += cross;
            weighted += cross * (current + next);
        }
        return std::abs(crossSum) > Epsilon ? weighted / (3.0f * crossSum) : Vector2::Zero();
    }

    bool FindCompactDestructionSplit2D(const std::vector<Vector2>& polygon,
                                       int negativePieceCount, int positivePieceCount,
                                       int branch, std::vector<Vector2>& bestNegative,
                                       std::vector<Vector2>& bestPositive)
    {
        constexpr int angleCandidateCount = 24;
        constexpr int offsetCandidateCount = 5;
        const int totalPieceCount = negativePieceCount + positivePieceCount;
        const float totalArea = PolygonArea2D(polygon);
        if (totalArea <= Epsilon || totalPieceCount <= 1) return false;

        const Vector2 centroid = PolygonCentroid2D(polygon);
        const float targetNegativeShare = static_cast<float>(negativePieceCount) /
                                          static_cast<float>(totalPieceCount);
        const float phase = 0.173f * static_cast<float>(branch) +
                            0.113f * static_cast<float>(totalPieceCount);
        const float preferredOffset = 0.045f *
            std::sin(1.37f * static_cast<float>(branch + totalPieceCount));
        float bestScore = FLT_MAX;

        for (int angleIndex = 0; angleIndex < angleCandidateCount; ++angleIndex)
        {
            const float angle = phase + Pi * static_cast<float>(angleIndex) /
                                        static_cast<float>(angleCandidateCount);
            const Vector2 normal{std::cos(angle), std::sin(angle)};
            const float projectionRadius = ProjectionRadius(polygon, centroid, normal);
            if (projectionRadius <= Epsilon) continue;

            for (int offsetIndex = 0; offsetIndex < offsetCandidateCount; ++offsetIndex)
            {
                const float offsetFraction = preferredOffset +
                    0.055f * static_cast<float>(offsetIndex - offsetCandidateCount / 2);
                const Vector2 linePoint = centroid + offsetFraction * projectionRadius * normal;
                std::vector<Vector2> negative;
                std::vector<Vector2> positive;
                if (!CutConvexPolygon2D(polygon, linePoint, normal, negative, positive)) continue;

                float negativeArea = PolygonArea2D(negative);
                float positiveArea = PolygonArea2D(positive);
                const float directError = std::abs(negativeArea / totalArea - targetNegativeShare);
                const float swappedError = std::abs(positiveArea / totalArea - targetNegativeShare);
                if (swappedError < directError)
                {
                    std::swap(negative, positive);
                    std::swap(negativeArea, positiveArea);
                }

                const float expectedNegativeArea = totalArea * targetNegativeShare;
                const float expectedPositiveArea = totalArea - expectedNegativeArea;
                if (negativeArea < 0.55f * expectedNegativeArea ||
                    positiveArea < 0.55f * expectedPositiveArea)
                {
                    continue;
                }

                const float areaError = std::abs(negativeArea / totalArea - targetNegativeShare);
                const float compactness = CompactnessPenalty(negative) + CompactnessPenalty(positive);
                const float offsetError = std::abs(offsetFraction - preferredOffset);
                const float score = 30.0f * areaError + 0.035f * compactness +
                                    0.15f * offsetError +
                                    1.5f * EdgeAlignmentPenalty(polygon, normal);
                if (score < bestScore)
                {
                    bestScore = score;
                    bestNegative = std::move(negative);
                    bestPositive = std::move(positive);
                }
            }
        }
        return bestScore < FLT_MAX;
    }
}
