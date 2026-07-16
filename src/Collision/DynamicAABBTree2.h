#pragma once

#include "Collision/AxisAlignedBox2.h"

#include <cstdint>
#include <vector>

namespace Riemann2D
{
    struct ProxyHandle
    {
        static constexpr std::uint32_t InvalidIndex = 0xffffffffu;

        std::uint32_t Index = InvalidIndex;
        std::uint32_t Generation = 0;

        constexpr bool IsValid() const
        {
            return Index != InvalidIndex && Generation != 0;
        }

        constexpr bool operator==(const ProxyHandle& other) const
        {
            return Index == other.Index && Generation == other.Generation;
        }

        constexpr bool operator!=(const ProxyHandle& other) const
        {
            return !(*this == other);
        }
    };

    class DynamicAABBTree2
    {
    public:
        using QueryCallback = bool (*)(ProxyHandle proxy, void* userData, void* context);

        static constexpr float DefaultFatMargin = 0.1f;
        static constexpr float DefaultDisplacementMultiplier = 2.0f;

        explicit DynamicAABBTree2(float fatMargin = DefaultFatMargin,
                               float displacementMultiplier = DefaultDisplacementMultiplier);

        void Clear();

        ProxyHandle CreateProxy(const AxisAlignedBox2& tightBounds, void* userData = nullptr);
        bool DestroyProxy(ProxyHandle proxy);
        bool MoveProxy(ProxyHandle proxy, const AxisAlignedBox2& tightBounds, const Vector2& displacement);

        void Query(const AxisAlignedBox2& bounds, QueryCallback callback, void* context = nullptr) const;
        std::vector<ProxyHandle> Query(const AxisAlignedBox2& bounds) const;

        bool IsValidProxy(ProxyHandle proxy) const;
        void* GetUserData(ProxyHandle proxy) const;
        const AxisAlignedBox2* GetFatBounds(ProxyHandle proxy) const;

        bool Validate() const;
        int GetHeight() const;
        std::uint32_t GetProxyCount() const;

    private:
        static constexpr std::uint32_t NullNode = ProxyHandle::InvalidIndex;

        struct Node
        {
            AxisAlignedBox2 Bounds{};
            void* UserData = nullptr;
            std::uint32_t Parent = NullNode;
            std::uint32_t Child1 = NullNode;
            std::uint32_t Child2 = NullNode;
            std::uint32_t NextFree = NullNode;
            std::uint32_t Generation = 1;
            int Height = -1;

            bool IsLeaf() const { return Child1 == NullNode; }
            bool IsFree() const { return Height < 0; }
        };

        std::uint32_t AllocateNode();
        void ReleaseNode(std::uint32_t nodeIndex);
        void GrowPool();

        void InsertLeaf(std::uint32_t leafIndex);
        void RemoveLeaf(std::uint32_t leafIndex);
        std::uint32_t Balance(std::uint32_t nodeIndex);

        AxisAlignedBox2 BuildFatBounds(const AxisAlignedBox2& tightBounds, const Vector2& displacement) const;
        bool ValidateNode(std::uint32_t nodeIndex, std::uint32_t parentIndex,
                          std::vector<bool>& visited, std::uint32_t& activeCount) const;

        std::vector<Node> Nodes;
        std::uint32_t Root = NullNode;
        std::uint32_t FreeList = NullNode;
        std::uint32_t ActiveNodeCount = 0;
        std::uint32_t ProxyCount = 0;
        float FatMargin = DefaultFatMargin;
        float DisplacementMultiplier = DefaultDisplacementMultiplier;
    };
}
