#pragma once

#include "Maths/Vector2.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Riemann2D::Render
{
    struct Color
    {
        float R = 1.0f;
        float G = 1.0f;
        float B = 1.0f;
        float A = 1.0f;
    };

    struct Vertex
    {
        Vector2 Position;
        Color Tint;
    };

    enum class PrimitiveTopology : std::uint8_t
    {
        Triangles,
        Lines
    };

    struct DrawPacket
    {
        std::string DebugName;
        PrimitiveTopology Topology = PrimitiveTopology::Triangles;
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
    };

    struct OrthographicCamera
    {
        Vector2 Center = Vector2::Zero();
        float VerticalSize = 20.0f;
        float NearPlane = -1.0f;
        float FarPlane = 1.0f;
    };

    struct FramePacket
    {
        std::uint64_t FrameId = 0;
        int ViewportWidth = 1280;
        int ViewportHeight = 720;
        int PanelWidth = 260;
        OrthographicCamera Camera;
        Color ClearColor{0.025f, 0.03f, 0.045f, 1.0f};
        std::vector<DrawPacket> WorldDraws;
        std::vector<DrawPacket> UserInterfaceDraws;
    };

    using SharedFrame = std::shared_ptr<const FramePacket>;
}
