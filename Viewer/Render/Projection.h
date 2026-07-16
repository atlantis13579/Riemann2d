#pragma once

#include "RenderTypes.h"

#include <algorithm>

namespace Riemann2D::Render
{
    inline Vector2 ProjectWorldToClip(const Vector2& position, const FramePacket& frame)
    {
        const float width = static_cast<float>(frame.ViewportWidth);
        const float height = static_cast<float>(frame.ViewportHeight);
        const float worldWidth = static_cast<float>(std::max(1, frame.ViewportWidth - frame.PanelWidth));
        const float aspect = worldWidth / std::max(1.0f, height);
        const float horizontalSize = std::max(0.001f, frame.Camera.VerticalSize * aspect);
        const float localX = (position.X - frame.Camera.Center.X) / (0.5f * horizontalSize);
        const float localY = (position.Y - frame.Camera.Center.Y) /
                             (0.5f * std::max(0.001f, frame.Camera.VerticalSize));
        const float pixelX = static_cast<float>(frame.PanelWidth) + (localX * 0.5f + 0.5f) * worldWidth;
        return {2.0f * pixelX / width - 1.0f, localY};
    }

    inline Vector2 ProjectInterfaceToClip(const Vector2& position, const FramePacket& frame)
    {
        return {
            2.0f * position.X / static_cast<float>(std::max(1, frame.ViewportWidth)) - 1.0f,
            1.0f - 2.0f * position.Y / static_cast<float>(std::max(1, frame.ViewportHeight))
        };
    }
}
