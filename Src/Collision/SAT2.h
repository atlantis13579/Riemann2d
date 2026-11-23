#pragma once

#include "Collision/AxisAlignedBox2.h"

#include <cstdint>
#include <vector>

namespace Riemann2D
{
    struct BroadPhaseProxy2
    {
        std::uint32_t Id = 0;
        AxisAlignedBox2 Bounds;
        bool IsStatic = false;
    };

    struct BroadPhasePair2
    {
        std::uint32_t A = 0;
        std::uint32_t B = 0;

        bool operator==(const BroadPhasePair2& other) const
        {
            return A == other.A && B == other.B;
        }
    };

    class SweepAndPruneBroadPhase2
    {
    public:
        void ComputePairs(const std::vector<BroadPhaseProxy2>& proxies,
                          std::vector<BroadPhasePair2>& pairs) const;
    };
}
