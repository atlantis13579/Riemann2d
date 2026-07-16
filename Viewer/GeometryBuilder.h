#pragma once

#include "Render/RenderTypes.h"

#include <vector>

namespace Riemann2D::Viewer
{
    void AddSolidBox(Render::DrawPacket& draw, const Vector2& center, const Vector2& halfExtent,
                     float angle, const Render::Color& color);
    void AddSolidCircle(Render::DrawPacket& draw, const Vector2& center, float radius,
                        const Render::Color& color, int segments = 32);
    void AddSolidCapsule(Render::DrawPacket& draw, const Vector2& center, float halfLength,
                         float radius, float angle, const Render::Color& color, int arcSegments = 12);
    void AddLine(Render::DrawPacket& draw, const Vector2& start, const Vector2& end,
                 const Render::Color& color);
    void AddSolidPolygon(Render::DrawPacket& draw, const std::vector<Vector2>& localVertices,
                         const Vector2& center, float angle, const Render::Color& color);
}
