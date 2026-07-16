#pragma once

#include "Maths/Vector2.h"

#include <algorithm>

namespace Riemann2D
{
    struct AxisAlignedBox2
    {
        Vector2 Minimum;
        Vector2 Maximum;

        bool IsValid() const
        {
            return Minimum.X <= Maximum.X && Minimum.Y <= Maximum.Y;
        }

        bool Overlaps(const AxisAlignedBox2& other) const
        {
            return Minimum.X <= other.Maximum.X && Maximum.X >= other.Minimum.X &&
                   Minimum.Y <= other.Maximum.Y && Maximum.Y >= other.Minimum.Y;
        }

        bool Contains(const AxisAlignedBox2& other) const
        {
            return Minimum.X <= other.Minimum.X && Minimum.Y <= other.Minimum.Y &&
                   Maximum.X >= other.Maximum.X && Maximum.Y >= other.Maximum.Y;
        }

        Vector2 Center() const { return 0.5f * (Minimum + Maximum); }
        Vector2 Extent() const { return 0.5f * (Maximum - Minimum); }
        float Perimeter() const
        {
            const Vector2 size = Maximum - Minimum;
            return 2.0f * (size.X + size.Y);
        }

        static AxisAlignedBox2 Union(const AxisAlignedBox2& a, const AxisAlignedBox2& b)
        {
            return {{std::min(a.Minimum.X, b.Minimum.X), std::min(a.Minimum.Y, b.Minimum.Y)},
                    {std::max(a.Maximum.X, b.Maximum.X), std::max(a.Maximum.Y, b.Maximum.Y)}};
        }

        void Expand(const Vector2& point)
        {
            Minimum.X = std::min(Minimum.X, point.X);
            Minimum.Y = std::min(Minimum.Y, point.Y);
            Maximum.X = std::max(Maximum.X, point.X);
            Maximum.Y = std::max(Maximum.Y, point.Y);
        }
    };
}
