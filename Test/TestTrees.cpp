#include "Test.h"

#include "Collision/StaticAABBTree2.h"
#include "Collision/DynamicAABBTree2.h"

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace Riemann2D;

R2_TEST("Static tree queries match brute force results")
{
    std::vector<StaticAABBTree2::Item> items;
    for (std::uint32_t y = 0; y < 10; ++y)
    {
        for (std::uint32_t x = 0; x < 10; ++x)
        {
            const Vector2 minimum{static_cast<float>(x) * 1.5f, static_cast<float>(y) * 1.25f};
            items.push_back({{minimum, minimum + Vector2{0.8f, 0.7f}}, y * 10 + x});
        }
    }

    StaticAABBTree2 tree;
    R2_REQUIRE(tree.Build(items));
    R2_REQUIRE(!tree.Empty());
    R2_REQUIRE(tree.GetNodeCount() == items.size() * 2 - 1);
    R2_REQUIRE(tree.GetHeight() <= 8);
    R2_REQUIRE(tree.Validate());

    const std::vector<AxisAlignedBox2> queries = {
        {{-1.0f, -1.0f}, {2.0f, 2.0f}},
        {{4.2f, 3.7f}, {10.1f, 8.4f}},
        {{13.0f, 10.0f}, {20.0f, 20.0f}},
        {{100.0f, 100.0f}, {101.0f, 101.0f}}
    };
    for (const AxisAlignedBox2& query : queries)
    {
        std::vector<std::uint32_t> expected;
        for (const StaticAABBTree2::Item& item : items)
        {
            if (item.Bounds.Overlaps(query)) expected.push_back(item.Payload);
        }
        std::vector<std::uint32_t> actual;
        tree.Query(query, actual);
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        R2_REQUIRE(actual == expected);
    }
}

R2_TEST("Static tree rejects invalid bounds and clears prior state")
{
    StaticAABBTree2 tree;
    R2_REQUIRE(tree.Build({{{{0.0f, 0.0f}, {1.0f, 1.0f}}, 7}}));
    R2_REQUIRE(!tree.Build({{{{2.0f, 0.0f}, {1.0f, 1.0f}}, 9}}));
    R2_REQUIRE(tree.Empty());
    R2_REQUIRE(tree.Validate());
}

R2_TEST("Dynamic tree inserts moves queries and removes proxies")
{
    DynamicAABBTree2 tree;
    const ProxyHandle first = tree.CreateProxy({{0.0f, 0.0f}, {1.0f, 1.0f}});
    const ProxyHandle second = tree.CreateProxy({{2.0f, 0.0f}, {3.0f, 1.0f}});
    const ProxyHandle third = tree.CreateProxy({{5.0f, 0.0f}, {6.0f, 1.0f}});
    R2_REQUIRE(first.IsValid() && second.IsValid() && third.IsValid());
    R2_REQUIRE(tree.GetProxyCount() == 3);
    R2_REQUIRE(tree.Validate());

    std::vector<ProxyHandle> hits = tree.Query({{-0.5f, -0.5f}, {3.2f, 1.5f}});
    R2_REQUIRE(hits.size() == 2);
    R2_REQUIRE(std::find(hits.begin(), hits.end(), first) != hits.end());
    R2_REQUIRE(std::find(hits.begin(), hits.end(), second) != hits.end());

    R2_REQUIRE(!tree.MoveProxy(first, {{0.02f, 0.0f}, {1.02f, 1.0f}}, {0.02f, 0.0f}));
    R2_REQUIRE(tree.MoveProxy(third, {{1.2f, 2.0f}, {2.2f, 3.0f}}, {-3.8f, 2.0f}));
    hits = tree.Query({{1.0f, 1.8f}, {2.4f, 3.2f}});
    R2_REQUIRE(std::find(hits.begin(), hits.end(), third) != hits.end());
    R2_REQUIRE(tree.Validate());

    R2_REQUIRE(tree.DestroyProxy(second));
    R2_REQUIRE(!tree.IsValidProxy(second));
    R2_REQUIRE(!tree.DestroyProxy(second));
    R2_REQUIRE(tree.GetProxyCount() == 2);
    R2_REQUIRE(tree.Validate());
}

R2_TEST("Dynamic tree generations invalidate recycled handles")
{
    DynamicAABBTree2 tree;
    const ProxyHandle oldHandle = tree.CreateProxy({{0.0f, 0.0f}, {1.0f, 1.0f}});
    R2_REQUIRE(tree.DestroyProxy(oldHandle));
    const ProxyHandle newHandle = tree.CreateProxy({{2.0f, 2.0f}, {3.0f, 3.0f}});
    R2_REQUIRE(newHandle.IsValid());
    R2_REQUIRE(newHandle != oldHandle);
    R2_REQUIRE(!tree.IsValidProxy(oldHandle));
    R2_REQUIRE(tree.IsValidProxy(newHandle));
}

R2_TEST("Dynamic tree randomized queries never miss tight bounds")
{
    DynamicAABBTree2 tree;
    std::vector<ProxyHandle> handles;
    std::vector<AxisAlignedBox2> bounds;
    for (int index = 0; index < 128; ++index)
    {
        const float x = static_cast<float>((index * 37) % 53) * 0.7f;
        const float y = static_cast<float>((index * 19) % 47) * 0.6f;
        const Vector2 extent{0.2f + 0.03f * static_cast<float>(index % 7),
                             0.25f + 0.02f * static_cast<float>(index % 5)};
        bounds.push_back({{x, y}, {x + extent.X, y + extent.Y}});
        handles.push_back(tree.CreateProxy(bounds.back()));
    }
    R2_REQUIRE(tree.Validate());

    for (int queryIndex = 0; queryIndex < 40; ++queryIndex)
    {
        const float x = static_cast<float>((queryIndex * 11) % 31) * 0.9f;
        const float y = static_cast<float>((queryIndex * 7) % 29) * 0.8f;
        const AxisAlignedBox2 query{{x, y}, {x + 4.0f, y + 3.0f}};
        const std::vector<ProxyHandle> hits = tree.Query(query);
        for (std::size_t index = 0; index < bounds.size(); ++index)
        {
            if (bounds[index].Overlaps(query))
            {
                R2_REQUIRE(std::find(hits.begin(), hits.end(), handles[index]) != hits.end());
            }
        }
    }

    for (std::size_t index = 0; index < handles.size(); index += 3)
    {
        const Vector2 displacement{5.0f, -2.0f};
        bounds[index].Minimum += displacement;
        bounds[index].Maximum += displacement;
        R2_REQUIRE(tree.MoveProxy(handles[index], bounds[index], displacement));
    }
    for (std::size_t index = 1; index < handles.size(); index += 4)
    {
        R2_REQUIRE(tree.DestroyProxy(handles[index]));
    }
    R2_REQUIRE(tree.Validate());
}
