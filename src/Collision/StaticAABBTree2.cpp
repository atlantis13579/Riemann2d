#include "Collision/StaticAABBTree2.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace Riemann2D
{
    namespace
    {
        bool IsFiniteAABB(const AxisAlignedBox2& bounds)
        {
            return bounds.IsValid() &&
                   std::isfinite(bounds.Minimum.X) && std::isfinite(bounds.Minimum.Y) &&
                   std::isfinite(bounds.Maximum.X) && std::isfinite(bounds.Maximum.Y);
        }

        AxisAlignedBox2 MergeBounds(const AxisAlignedBox2& first, const AxisAlignedBox2& second)
        {
            return {
                {std::min(first.Minimum.X, second.Minimum.X), std::min(first.Minimum.Y, second.Minimum.Y)},
                {std::max(first.Maximum.X, second.Maximum.X), std::max(first.Maximum.Y, second.Maximum.Y)}};
        }

        bool Contains(const AxisAlignedBox2& outer, const AxisAlignedBox2& inner)
        {
            return outer.Minimum.X <= inner.Minimum.X && outer.Minimum.Y <= inner.Minimum.Y &&
                   outer.Maximum.X >= inner.Maximum.X && outer.Maximum.Y >= inner.Maximum.Y;
        }
    }

    bool StaticAABBTree2::Build(const std::vector<Item>& items)
    {
        Clear();
        if (items.empty())
        {
            return true;
        }

        const std::size_t maximumItemCount = static_cast<std::size_t>(InvalidNode) / 2 + 1;
        if (items.size() > maximumItemCount)
        {
            return false;
        }

        std::vector<BuildItem> buildItems;
        buildItems.reserve(items.size());
        for (const Item& item : items)
        {
            if (!IsFiniteAABB(item.Bounds))
            {
                return false;
            }

            buildItems.push_back({
                item,
                {item.Bounds.Minimum.X * 0.5f + item.Bounds.Maximum.X * 0.5f,
                 item.Bounds.Minimum.Y * 0.5f + item.Bounds.Maximum.Y * 0.5f}});
        }

        Nodes.reserve(items.size() * 2 - 1);
        BuildRange(buildItems, 0, buildItems.size());
        return true;
    }

    std::uint32_t StaticAABBTree2::BuildRange(
        std::vector<BuildItem>& items, std::size_t begin, std::size_t end)
    {
        const std::uint32_t nodeIndex = static_cast<std::uint32_t>(Nodes.size());
        Nodes.emplace_back();

        if (end - begin == 1)
        {
            Nodes[nodeIndex].Bounds = items[begin].Source.Bounds;
            Nodes[nodeIndex].Payload = items[begin].Source.Payload;
            return nodeIndex;
        }

        Vector2 centerMinimum = items[begin].Center;
        Vector2 centerMaximum = items[begin].Center;
        for (std::size_t index = begin + 1; index < end; ++index)
        {
            centerMinimum.X = std::min(centerMinimum.X, items[index].Center.X);
            centerMinimum.Y = std::min(centerMinimum.Y, items[index].Center.Y);
            centerMaximum.X = std::max(centerMaximum.X, items[index].Center.X);
            centerMaximum.Y = std::max(centerMaximum.Y, items[index].Center.Y);
        }

        const Vector2 centerExtent = centerMaximum - centerMinimum;
        const bool splitOnX = centerExtent.X >= centerExtent.Y;
        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(
            items.begin() + static_cast<std::ptrdiff_t>(begin),
            items.begin() + static_cast<std::ptrdiff_t>(middle),
            items.begin() + static_cast<std::ptrdiff_t>(end),
            [splitOnX](const BuildItem& first, const BuildItem& second)
            {
                const float firstCoordinate = splitOnX ? first.Center.X : first.Center.Y;
                const float secondCoordinate = splitOnX ? second.Center.X : second.Center.Y;
                if (firstCoordinate != secondCoordinate)
                {
                    return firstCoordinate < secondCoordinate;
                }
                return first.Source.Payload < second.Source.Payload;
            });

        const std::uint32_t left = BuildRange(items, begin, middle);
        const std::uint32_t right = BuildRange(items, middle, end);
        Nodes[nodeIndex].Left = left;
        Nodes[nodeIndex].Right = right;
        Nodes[nodeIndex].Bounds = MergeBounds(Nodes[left].Bounds, Nodes[right].Bounds);
        return nodeIndex;
    }

    void StaticAABBTree2::Query(
        const AxisAlignedBox2& queryBounds, std::vector<std::uint32_t>& output) const
    {
        output.clear();
        if (Nodes.empty() || !IsFiniteAABB(queryBounds))
        {
            return;
        }

        std::vector<std::uint32_t> pending;
        pending.push_back(0);
        while (!pending.empty())
        {
            const std::uint32_t nodeIndex = pending.back();
            pending.pop_back();
            const Node& node = Nodes[nodeIndex];
            if (!node.Bounds.Overlaps(queryBounds))
            {
                continue;
            }

            if (node.IsLeaf())
            {
                output.push_back(node.Payload);
            }
            else
            {
                pending.push_back(node.Right);
                pending.push_back(node.Left);
            }
        }
    }

    void StaticAABBTree2::Clear()
    {
        Nodes.clear();
    }

    bool StaticAABBTree2::Empty() const
    {
        return Nodes.empty();
    }

    std::size_t StaticAABBTree2::GetNodeCount() const
    {
        return Nodes.size();
    }

    std::size_t StaticAABBTree2::GetHeight() const
    {
        return Nodes.empty() ? 0 : ComputeHeight(0);
    }

    std::size_t StaticAABBTree2::ComputeHeight(std::uint32_t nodeIndex) const
    {
        const Node& node = Nodes[nodeIndex];
        if (node.IsLeaf())
        {
            return 1;
        }

        return 1 + std::max(ComputeHeight(node.Left), ComputeHeight(node.Right));
    }

    bool StaticAABBTree2::Validate() const
    {
        if (Nodes.empty())
        {
            return true;
        }

        std::vector<std::uint8_t> visited(Nodes.size(), 0);
        if (!ValidateNode(0, visited))
        {
            return false;
        }

        return std::all_of(visited.begin(), visited.end(), [](std::uint8_t value) { return value != 0; });
    }

    bool StaticAABBTree2::ValidateNode(
        std::uint32_t nodeIndex, std::vector<std::uint8_t>& visited) const
    {
        if (nodeIndex >= Nodes.size() || visited[nodeIndex] != 0)
        {
            return false;
        }

        visited[nodeIndex] = 1;
        const Node& node = Nodes[nodeIndex];
        if (!IsFiniteAABB(node.Bounds))
        {
            return false;
        }

        if (node.IsLeaf())
        {
            return node.Right == InvalidNode;
        }

        if (node.Right == InvalidNode || node.Left <= nodeIndex || node.Right <= nodeIndex ||
            node.Left >= Nodes.size() || node.Right >= Nodes.size())
        {
            return false;
        }

        const Node& left = Nodes[node.Left];
        const Node& right = Nodes[node.Right];
        if (!Contains(node.Bounds, left.Bounds) || !Contains(node.Bounds, right.Bounds))
        {
            return false;
        }

        return ValidateNode(node.Left, visited) && ValidateNode(node.Right, visited);
    }
}
