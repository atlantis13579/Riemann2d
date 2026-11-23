#include "Collision/DynamicAABBTree2.h"

#include <algorithm>

namespace Riemann2D
{
    namespace
    {
        AxisAlignedBox2 Combine(const AxisAlignedBox2& first, const AxisAlignedBox2& second)
        {
            return {
                {std::min(first.Minimum.X, second.Minimum.X), std::min(first.Minimum.Y, second.Minimum.Y)},
                {std::max(first.Maximum.X, second.Maximum.X), std::max(first.Maximum.Y, second.Maximum.Y)}};
        }

        float Perimeter(const AxisAlignedBox2& bounds)
        {
            const Vector2 size = bounds.Maximum - bounds.Minimum;
            return 2.0f * (size.X + size.Y);
        }

        bool Contains(const AxisAlignedBox2& outer, const AxisAlignedBox2& inner)
        {
            return outer.Minimum.X <= inner.Minimum.X && outer.Minimum.Y <= inner.Minimum.Y &&
                   outer.Maximum.X >= inner.Maximum.X && outer.Maximum.Y >= inner.Maximum.Y;
        }

        bool BoundsEqual(const AxisAlignedBox2& first, const AxisAlignedBox2& second)
        {
            return first.Minimum.X == second.Minimum.X && first.Minimum.Y == second.Minimum.Y &&
                   first.Maximum.X == second.Maximum.X && first.Maximum.Y == second.Maximum.Y;
        }

        std::uint32_t NextGeneration(std::uint32_t generation)
        {
            ++generation;
            return generation == 0 ? 1 : generation;
        }
    }

    DynamicAABBTree2::DynamicAABBTree2(float fatMargin, float displacementMultiplier)
        : FatMargin(std::max(0.0f, fatMargin))
        , DisplacementMultiplier(std::max(0.0f, displacementMultiplier))
    {
        GrowPool();
    }

    void DynamicAABBTree2::Clear()
    {
        Root = NullNode;
        FreeList = Nodes.empty() ? NullNode : 0;
        ActiveNodeCount = 0;
        ProxyCount = 0;

        for (std::uint32_t i = 0; i < Nodes.size(); ++i)
        {
            Node& node = Nodes[i];
            node.UserData = nullptr;
            node.Parent = NullNode;
            node.Child1 = NullNode;
            node.Child2 = NullNode;
            node.NextFree = i + 1 < Nodes.size() ? i + 1 : NullNode;
            node.Generation = NextGeneration(node.Generation);
            node.Height = -1;
        }
    }

    ProxyHandle DynamicAABBTree2::CreateProxy(const AxisAlignedBox2& tightBounds, void* userData)
    {
        if (!tightBounds.IsValid())
        {
            return {};
        }

        const std::uint32_t leafIndex = AllocateNode();
        Node& leaf = Nodes[leafIndex];
        leaf.Bounds = BuildFatBounds(tightBounds, Vector2::Zero());
        leaf.UserData = userData;
        leaf.Height = 0;
        InsertLeaf(leafIndex);
        ++ProxyCount;
        // InsertLeaf may grow the node vector and invalidate the earlier reference.
        return {leafIndex, Nodes[leafIndex].Generation};
    }

    bool DynamicAABBTree2::DestroyProxy(ProxyHandle proxy)
    {
        if (!IsValidProxy(proxy))
        {
            return false;
        }

        RemoveLeaf(proxy.Index);
        ReleaseNode(proxy.Index);
        --ProxyCount;
        return true;
    }

    bool DynamicAABBTree2::MoveProxy(ProxyHandle proxy, const AxisAlignedBox2& tightBounds, const Vector2& displacement)
    {
        if (!IsValidProxy(proxy) || !tightBounds.IsValid())
        {
            return false;
        }

        Node& leaf = Nodes[proxy.Index];
        if (Contains(leaf.Bounds, tightBounds))
        {
            return false;
        }

        RemoveLeaf(proxy.Index);
        leaf.Bounds = BuildFatBounds(tightBounds, displacement);
        InsertLeaf(proxy.Index);
        return true;
    }

    void DynamicAABBTree2::Query(const AxisAlignedBox2& bounds, QueryCallback callback, void* context) const
    {
        if (Root == NullNode || callback == nullptr || !bounds.IsValid())
        {
            return;
        }

        std::vector<std::uint32_t> stack;
        stack.push_back(Root);
        while (!stack.empty())
        {
            const std::uint32_t nodeIndex = stack.back();
            stack.pop_back();
            const Node& node = Nodes[nodeIndex];
            if (!node.Bounds.Overlaps(bounds))
            {
                continue;
            }

            if (node.IsLeaf())
            {
                if (!callback({nodeIndex, node.Generation}, node.UserData, context))
                {
                    return;
                }
            }
            else
            {
                stack.push_back(node.Child1);
                stack.push_back(node.Child2);
            }
        }
    }

    std::vector<ProxyHandle> DynamicAABBTree2::Query(const AxisAlignedBox2& bounds) const
    {
        std::vector<ProxyHandle> result;
        struct Collector
        {
            static bool Add(ProxyHandle proxy, void*, void* context)
            {
                static_cast<std::vector<ProxyHandle>*>(context)->push_back(proxy);
                return true;
            }
        };
        Query(bounds, &Collector::Add, &result);
        return result;
    }

    bool DynamicAABBTree2::IsValidProxy(ProxyHandle proxy) const
    {
        if (!proxy.IsValid() || proxy.Index >= Nodes.size())
        {
            return false;
        }
        const Node& node = Nodes[proxy.Index];
        return !node.IsFree() && node.IsLeaf() && node.Generation == proxy.Generation;
    }

    void* DynamicAABBTree2::GetUserData(ProxyHandle proxy) const
    {
        return IsValidProxy(proxy) ? Nodes[proxy.Index].UserData : nullptr;
    }

    const AxisAlignedBox2* DynamicAABBTree2::GetFatBounds(ProxyHandle proxy) const
    {
        return IsValidProxy(proxy) ? &Nodes[proxy.Index].Bounds : nullptr;
    }

    int DynamicAABBTree2::GetHeight() const
    {
        return Root == NullNode ? 0 : Nodes[Root].Height;
    }

    std::uint32_t DynamicAABBTree2::GetProxyCount() const
    {
        return ProxyCount;
    }

    std::uint32_t DynamicAABBTree2::AllocateNode()
    {
        if (FreeList == NullNode)
        {
            GrowPool();
        }

        const std::uint32_t nodeIndex = FreeList;
        Node& node = Nodes[nodeIndex];
        FreeList = node.NextFree;
        node.Parent = NullNode;
        node.Child1 = NullNode;
        node.Child2 = NullNode;
        node.NextFree = NullNode;
        node.UserData = nullptr;
        node.Height = 0;
        ++ActiveNodeCount;
        return nodeIndex;
    }

    void DynamicAABBTree2::ReleaseNode(std::uint32_t nodeIndex)
    {
        Node& node = Nodes[nodeIndex];
        node.UserData = nullptr;
        node.Parent = NullNode;
        node.Child1 = NullNode;
        node.Child2 = NullNode;
        node.NextFree = FreeList;
        node.Generation = NextGeneration(node.Generation);
        node.Height = -1;
        FreeList = nodeIndex;
        --ActiveNodeCount;
    }

    void DynamicAABBTree2::GrowPool()
    {
        const std::uint32_t oldSize = static_cast<std::uint32_t>(Nodes.size());
        const std::uint32_t newSize = oldSize == 0 ? 16u : oldSize * 2u;
        Nodes.resize(newSize);
        for (std::uint32_t i = oldSize; i < newSize; ++i)
        {
            Nodes[i].Height = -1;
            Nodes[i].NextFree = i + 1 < newSize ? i + 1 : FreeList;
        }
        FreeList = oldSize;
    }

    AxisAlignedBox2 DynamicAABBTree2::BuildFatBounds(const AxisAlignedBox2& tightBounds, const Vector2& displacement) const
    {
        AxisAlignedBox2 result = tightBounds;
        result.Minimum -= Vector2(FatMargin, FatMargin);
        result.Maximum += Vector2(FatMargin, FatMargin);

        const Vector2 prediction = displacement * DisplacementMultiplier;
        if (prediction.X < 0.0f)
            result.Minimum.X += prediction.X;
        else
            result.Maximum.X += prediction.X;
        if (prediction.Y < 0.0f)
            result.Minimum.Y += prediction.Y;
        else
            result.Maximum.Y += prediction.Y;
        return result;
    }

    void DynamicAABBTree2::InsertLeaf(std::uint32_t leafIndex)
    {
        if (Root == NullNode)
        {
            Root = leafIndex;
            Nodes[Root].Parent = NullNode;
            return;
        }

        const AxisAlignedBox2 leafBounds = Nodes[leafIndex].Bounds;
        std::uint32_t sibling = Root;
        while (!Nodes[sibling].IsLeaf())
        {
            const Node& branch = Nodes[sibling];
            const float branchPerimeter = Perimeter(branch.Bounds);
            const AxisAlignedBox2 combined = Combine(branch.Bounds, leafBounds);
            const float combinedPerimeter = Perimeter(combined);
            const float parentCost = 2.0f * combinedPerimeter;
            const float inheritedCost = 2.0f * (combinedPerimeter - branchPerimeter);

            const Node& child1 = Nodes[branch.Child1];
            const Node& child2 = Nodes[branch.Child2];
            const AxisAlignedBox2 combined1 = Combine(child1.Bounds, leafBounds);
            const AxisAlignedBox2 combined2 = Combine(child2.Bounds, leafBounds);
            const float cost1 = (child1.IsLeaf() ? Perimeter(combined1)
                                                : Perimeter(combined1) - Perimeter(child1.Bounds)) + inheritedCost;
            const float cost2 = (child2.IsLeaf() ? Perimeter(combined2)
                                                : Perimeter(combined2) - Perimeter(child2.Bounds)) + inheritedCost;

            if (parentCost < cost1 && parentCost < cost2)
            {
                break;
            }
            sibling = cost1 < cost2 ? branch.Child1 : branch.Child2;
        }

        const std::uint32_t oldParent = Nodes[sibling].Parent;
        const std::uint32_t newParent = AllocateNode();
        Nodes[newParent].Parent = oldParent;
        Nodes[newParent].Bounds = Combine(leafBounds, Nodes[sibling].Bounds);
        Nodes[newParent].Height = Nodes[sibling].Height + 1;
        Nodes[newParent].Child1 = sibling;
        Nodes[newParent].Child2 = leafIndex;
        Nodes[sibling].Parent = newParent;
        Nodes[leafIndex].Parent = newParent;

        if (oldParent == NullNode)
        {
            Root = newParent;
        }
        else if (Nodes[oldParent].Child1 == sibling)
        {
            Nodes[oldParent].Child1 = newParent;
        }
        else
        {
            Nodes[oldParent].Child2 = newParent;
        }

        std::uint32_t current = Nodes[leafIndex].Parent;
        while (current != NullNode)
        {
            current = Balance(current);
            Node& node = Nodes[current];
            const Node& child1 = Nodes[node.Child1];
            const Node& child2 = Nodes[node.Child2];
            node.Height = 1 + std::max(child1.Height, child2.Height);
            node.Bounds = Combine(child1.Bounds, child2.Bounds);
            current = node.Parent;
        }
    }

    void DynamicAABBTree2::RemoveLeaf(std::uint32_t leafIndex)
    {
        if (leafIndex == Root)
        {
            Root = NullNode;
            Nodes[leafIndex].Parent = NullNode;
            return;
        }

        const std::uint32_t parent = Nodes[leafIndex].Parent;
        const std::uint32_t grandParent = Nodes[parent].Parent;
        const std::uint32_t sibling = Nodes[parent].Child1 == leafIndex
                                          ? Nodes[parent].Child2
                                          : Nodes[parent].Child1;

        if (grandParent == NullNode)
        {
            Root = sibling;
            Nodes[sibling].Parent = NullNode;
            ReleaseNode(parent);
        }
        else
        {
            if (Nodes[grandParent].Child1 == parent)
                Nodes[grandParent].Child1 = sibling;
            else
                Nodes[grandParent].Child2 = sibling;
            Nodes[sibling].Parent = grandParent;
            ReleaseNode(parent);

            std::uint32_t current = grandParent;
            while (current != NullNode)
            {
                current = Balance(current);
                Node& node = Nodes[current];
                const Node& child1 = Nodes[node.Child1];
                const Node& child2 = Nodes[node.Child2];
                node.Bounds = Combine(child1.Bounds, child2.Bounds);
                node.Height = 1 + std::max(child1.Height, child2.Height);
                current = node.Parent;
            }
        }
        Nodes[leafIndex].Parent = NullNode;
    }

    std::uint32_t DynamicAABBTree2::Balance(std::uint32_t nodeIndex)
    {
        Node& node = Nodes[nodeIndex];
        if (node.IsLeaf() || node.Height < 2)
        {
            return nodeIndex;
        }

        const std::uint32_t leftIndex = node.Child1;
        const std::uint32_t rightIndex = node.Child2;
        Node& left = Nodes[leftIndex];
        Node& right = Nodes[rightIndex];
        const int imbalance = right.Height - left.Height;

        if (imbalance > 1)
        {
            const std::uint32_t rightLeftIndex = right.Child1;
            const std::uint32_t rightRightIndex = right.Child2;
            Node& rightLeft = Nodes[rightLeftIndex];
            Node& rightRight = Nodes[rightRightIndex];

            right.Child1 = nodeIndex;
            right.Parent = node.Parent;
            node.Parent = rightIndex;
            if (right.Parent == NullNode)
                Root = rightIndex;
            else if (Nodes[right.Parent].Child1 == nodeIndex)
                Nodes[right.Parent].Child1 = rightIndex;
            else
                Nodes[right.Parent].Child2 = rightIndex;

            if (rightLeft.Height > rightRight.Height)
            {
                right.Child2 = rightLeftIndex;
                node.Child2 = rightRightIndex;
                rightRight.Parent = nodeIndex;
                rightLeft.Parent = rightIndex;
                node.Bounds = Combine(left.Bounds, rightRight.Bounds);
                right.Bounds = Combine(node.Bounds, rightLeft.Bounds);
                node.Height = 1 + std::max(left.Height, rightRight.Height);
                right.Height = 1 + std::max(node.Height, rightLeft.Height);
            }
            else
            {
                right.Child2 = rightRightIndex;
                node.Child2 = rightLeftIndex;
                rightLeft.Parent = nodeIndex;
                rightRight.Parent = rightIndex;
                node.Bounds = Combine(left.Bounds, rightLeft.Bounds);
                right.Bounds = Combine(node.Bounds, rightRight.Bounds);
                node.Height = 1 + std::max(left.Height, rightLeft.Height);
                right.Height = 1 + std::max(node.Height, rightRight.Height);
            }
            return rightIndex;
        }

        if (imbalance < -1)
        {
            const std::uint32_t leftLeftIndex = left.Child1;
            const std::uint32_t leftRightIndex = left.Child2;
            Node& leftLeft = Nodes[leftLeftIndex];
            Node& leftRight = Nodes[leftRightIndex];

            left.Child1 = nodeIndex;
            left.Parent = node.Parent;
            node.Parent = leftIndex;
            if (left.Parent == NullNode)
                Root = leftIndex;
            else if (Nodes[left.Parent].Child1 == nodeIndex)
                Nodes[left.Parent].Child1 = leftIndex;
            else
                Nodes[left.Parent].Child2 = leftIndex;

            if (leftLeft.Height > leftRight.Height)
            {
                left.Child2 = leftLeftIndex;
                node.Child1 = leftRightIndex;
                leftRight.Parent = nodeIndex;
                leftLeft.Parent = leftIndex;
                node.Bounds = Combine(right.Bounds, leftRight.Bounds);
                left.Bounds = Combine(node.Bounds, leftLeft.Bounds);
                node.Height = 1 + std::max(right.Height, leftRight.Height);
                left.Height = 1 + std::max(node.Height, leftLeft.Height);
            }
            else
            {
                left.Child2 = leftRightIndex;
                node.Child1 = leftLeftIndex;
                leftLeft.Parent = nodeIndex;
                leftRight.Parent = leftIndex;
                node.Bounds = Combine(right.Bounds, leftLeft.Bounds);
                left.Bounds = Combine(node.Bounds, leftRight.Bounds);
                node.Height = 1 + std::max(right.Height, leftLeft.Height);
                left.Height = 1 + std::max(node.Height, leftRight.Height);
            }
            return leftIndex;
        }

        return nodeIndex;
    }

    bool DynamicAABBTree2::Validate() const
    {
        if (Root == NullNode)
        {
            return ActiveNodeCount == 0 && ProxyCount == 0;
        }
        if (Root >= Nodes.size() || Nodes[Root].Parent != NullNode)
        {
            return false;
        }

        std::vector<bool> visited(Nodes.size(), false);
        std::uint32_t activeCount = 0;
        if (!ValidateNode(Root, NullNode, visited, activeCount) || activeCount != ActiveNodeCount ||
            ProxyCount != (activeCount + 1) / 2)
        {
            return false;
        }

        std::uint32_t freeCount = 0;
        std::uint32_t freeNode = FreeList;
        while (freeNode != NullNode)
        {
            if (freeNode >= Nodes.size() || visited[freeNode] || !Nodes[freeNode].IsFree())
            {
                return false;
            }
            visited[freeNode] = true;
            ++freeCount;
            freeNode = Nodes[freeNode].NextFree;
        }
        return activeCount + freeCount == Nodes.size();
    }

    bool DynamicAABBTree2::ValidateNode(std::uint32_t nodeIndex, std::uint32_t parentIndex,
                                     std::vector<bool>& visited, std::uint32_t& activeCount) const
    {
        if (nodeIndex >= Nodes.size() || visited[nodeIndex])
        {
            return false;
        }
        visited[nodeIndex] = true;
        ++activeCount;
        const Node& node = Nodes[nodeIndex];
        if (node.IsFree() || node.Parent != parentIndex || !node.Bounds.IsValid())
        {
            return false;
        }
        if (node.IsLeaf())
        {
            return node.Child2 == NullNode && node.Height == 0;
        }
        if (node.Child2 == NullNode || node.Child1 == node.Child2)
        {
            return false;
        }
        if (node.Child1 >= Nodes.size() || node.Child2 >= Nodes.size())
        {
            return false;
        }

        const Node& child1 = Nodes[node.Child1];
        const Node& child2 = Nodes[node.Child2];
        if (node.Height != 1 + std::max(child1.Height, child2.Height) ||
            !BoundsEqual(node.Bounds, Combine(child1.Bounds, child2.Bounds)))
        {
            return false;
        }
        return ValidateNode(node.Child1, nodeIndex, visited, activeCount) &&
               ValidateNode(node.Child2, nodeIndex, visited, activeCount);
    }
}
