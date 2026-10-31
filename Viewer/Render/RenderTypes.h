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
        Vector2 TexCoord = Vector2::Zero();
    };

    enum class PrimitiveTopology : std::uint8_t
    {
        Triangles,
        Lines
    };

    enum class FieldMaterial : std::uint8_t
    {
        Water = 1,
        Smoke = 2
    };

    struct ScalarField
    {
        int Width = 0;
        int Height = 0;
        FieldMaterial Material = FieldMaterial::Water;
        float Rotation = 0.0f;
        std::vector<float> Values;
        bool IsValid() const
        {
            return Width > 0 && Height > 0 && Width <= 4096 && Height <= 4096 &&
                Values.size() == static_cast<std::size_t>(Width) * Height;
        }
    };

    struct DrawPacket
    {
        std::string DebugName;
        PrimitiveTopology Topology = PrimitiveTopology::Triangles;
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
        std::shared_ptr<const ScalarField> Field;
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
        // Optional deterministic framebuffer capture for offline render verification.
        std::string CapturePath;
    };

    using SharedFrame = std::shared_ptr<const FramePacket>;
}
