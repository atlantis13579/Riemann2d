#pragma once

#include "Collision/AxisAlignedBox2.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace Riemann2D
{
    class StaticAABBTree2
    {
    public:
        struct Item
        {
            AxisAlignedBox2 Bounds;
            std::uint32_t Payload = 0;
        };

        // Replaces the current tree. Invalid input leaves the tree empty.
        bool Build(const std::vector<Item>& items);

        // Replaces output with all payloads whose bounds overlap queryBounds.
        void Query(const AxisAlignedBox2& queryBounds, std::vector<std::uint32_t>& output) const;

        void Clear();
        bool Empty() const;
        std::size_t GetNodeCount() const;
        std::size_t GetHeight() const;
        bool Validate() const;

    private:
        static constexpr std::uint32_t InvalidNode = std::numeric_limits<std::uint32_t>::max();

        struct Node
        {
            AxisAlignedBox2 Bounds;
            std::uint32_t Left = InvalidNode;
            std::uint32_t Right = InvalidNode;
            std::uint32_t Payload = 0;

            bool IsLeaf() const
            {
                return Left == InvalidNode;
            }
        };

        struct BuildItem
        {
            Item Source;
            Vector2 Center;
        };

        std::uint32_t BuildRange(std::vector<BuildItem>& items, std::size_t begin, std::size_t end);
        std::size_t ComputeHeight(std::uint32_t nodeIndex) const;
        bool ValidateNode(std::uint32_t nodeIndex, std::vector<std::uint8_t>& visited) const;

        std::vector<Node> Nodes;
    };
}
