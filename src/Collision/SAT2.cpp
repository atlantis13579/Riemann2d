#include "SAT2.h"

#include <algorithm>
#include <cstddef>
#include <numeric>

namespace Riemann2D
{
    void SweepAndPruneBroadPhase2::ComputePairs(
        const std::vector<BroadPhaseProxy2>& proxies,
        std::vector<BroadPhasePair2>& pairs) const
    {
        pairs.clear();
        std::vector<std::size_t> order(proxies.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(), [&proxies](std::size_t left, std::size_t right)
        {
            const BroadPhaseProxy2& a = proxies[left];
            const BroadPhaseProxy2& b = proxies[right];
            return a.Bounds.Minimum.X < b.Bounds.Minimum.X ||
                   (a.Bounds.Minimum.X == b.Bounds.Minimum.X && a.Id < b.Id);
        });

        std::vector<std::size_t> active;
        active.reserve(proxies.size());
        for (const std::size_t proxyIndex : order)
        {
            const BroadPhaseProxy2& current = proxies[proxyIndex];
            if (!current.Bounds.IsValid())
            {
                continue;
            }
            active.erase(std::remove_if(active.begin(), active.end(),
                [&proxies, &current](std::size_t index)
                {
                    return proxies[index].Bounds.Maximum.X < current.Bounds.Minimum.X;
                }), active.end());

            for (const std::size_t otherIndex : active)
            {
                const BroadPhaseProxy2& other = proxies[otherIndex];
                if ((current.IsStatic && other.IsStatic) ||
                    current.Bounds.Minimum.Y > other.Bounds.Maximum.Y ||
                    current.Bounds.Maximum.Y < other.Bounds.Minimum.Y)
                {
                    continue;
                }
                pairs.push_back({std::min(current.Id, other.Id), std::max(current.Id, other.Id)});
            }
            active.push_back(proxyIndex);
        }

        std::sort(pairs.begin(), pairs.end(), [](const BroadPhasePair2& left,
                                                 const BroadPhasePair2& right)
        {
            return left.A < right.A || (left.A == right.A && left.B < right.B);
        });
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    }
}
