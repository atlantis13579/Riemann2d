#include "GeometryBuilder.h"

#include "Maths/Maths.h"

#include <array>
#include <cmath>

namespace Riemann2D::Viewer
{
    namespace
    {
        void AddTriangle(Render::DrawPacket& draw, const Vector2& a, const Vector2& b, const Vector2& c,
                         const Render::Color& color)
        {
            const std::uint32_t first = static_cast<std::uint32_t>(draw.Vertices.size());
            draw.Vertices.push_back({a, color});
            draw.Vertices.push_back({b, color});
            draw.Vertices.push_back({c, color});
            draw.Indices.insert(draw.Indices.end(), {first, first + 1, first + 2});
        }
    }

    void AddSolidBox(Render::DrawPacket& draw, const Vector2& center, const Vector2& halfExtent,
                     float angle, const Render::Color& color)
    {
        const std::array<Vector2, 4> local = {
            Vector2{-halfExtent.X, -halfExtent.Y}, Vector2{halfExtent.X, -halfExtent.Y},
            Vector2{halfExtent.X, halfExtent.Y}, Vector2{-halfExtent.X, halfExtent.Y}
        };
        std::array<Vector2, 4> world{};
        for (std::size_t index = 0; index < world.size(); ++index)
        {
            world[index] = center + local[index].Rotated(angle);
        }
        AddTriangle(draw, world[0], world[1], world[2], color);
        AddTriangle(draw, world[0], world[2], world[3], color);
    }

    void AddSolidCircle(Render::DrawPacket& draw, const Vector2& center, float radius,
                        const Render::Color& color, int segments)
    {
        for (int index = 0; index < segments; ++index)
        {
            const float angle0 = 2.0f * Pi * static_cast<float>(index) / static_cast<float>(segments);
            const float angle1 = 2.0f * Pi * static_cast<float>(index + 1) / static_cast<float>(segments);
            AddTriangle(draw, center,
                        center + radius * Vector2{std::cos(angle0), std::sin(angle0)},
                        center + radius * Vector2{std::cos(angle1), std::sin(angle1)}, color);
        }
    }

    void AddSolidCapsule(Render::DrawPacket& draw, const Vector2& center, float halfLength,
                         float radius, float angle, const Render::Color& color, int arcSegments)
    {
        AddSolidBox(draw, center, {halfLength, radius}, angle, color);
        const Vector2 axis = Vector2::UnitX().Rotated(angle);
        AddSolidCircle(draw, center - axis * halfLength, radius, color, arcSegments * 2);
        AddSolidCircle(draw, center + axis * halfLength, radius, color, arcSegments * 2);
    }

    void AddLine(Render::DrawPacket& draw, const Vector2& start, const Vector2& end,
                 const Render::Color& color)
    {
        const std::uint32_t first = static_cast<std::uint32_t>(draw.Vertices.size());
        draw.Vertices.push_back({start, color});
        draw.Vertices.push_back({end, color});
        draw.Indices.insert(draw.Indices.end(), {first, first + 1});
    }

    void AddSolidPolygon(Render::DrawPacket& draw, const std::vector<Vector2>& localVertices,
                         const Vector2& center, float angle, const Render::Color& color)
    {
        if (localVertices.size() < 3)
        {
            return;
        }
        const Vector2 first = center + localVertices[0].Rotated(angle);
        for (std::size_t index = 1; index + 1 < localVertices.size(); ++index)
        {
            AddTriangle(draw, first, center + localVertices[index].Rotated(angle),
                        center + localVertices[index + 1].Rotated(angle), color);
        }
    }
}
