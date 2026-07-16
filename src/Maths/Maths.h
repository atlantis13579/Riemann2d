#pragma once

#include <algorithm>
#include <cmath>

namespace Riemann2D
{
    constexpr float Pi = 3.14159265358979323846f;
    constexpr float Epsilon = 1.0e-6f;

    template <typename T>
    constexpr T Clamp(T value, T minimum, T maximum)
    {
        return std::min(std::max(value, minimum), maximum);
    }

    inline bool NearlyEqual(float a, float b, float tolerance = Epsilon)
    {
        return std::abs(a - b) <= tolerance;
    }
}
