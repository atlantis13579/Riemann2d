#include "FluidSurfaceBuilder.h"

#include "Maths/Maths.h"

#include <algorithm>
#include <cmath>

namespace Riemann2D::Viewer
{
    namespace
    {
        void BuildFieldQuad(Render::DrawPacket& draw, const Vector2& minimum,
                            const Vector2& maximum, const Transform2& transform = {})
        {
            draw.Vertices = {
                {transform.TransformPoint(minimum), {}, {0.0f, 0.0f}},
                {transform.TransformPoint({maximum.X, minimum.Y}), {}, {1.0f, 0.0f}},
                {transform.TransformPoint(maximum), {}, {1.0f, 1.0f}},
                {transform.TransformPoint({minimum.X, maximum.Y}), {}, {0.0f, 1.0f}}
            };
            draw.Indices = {0, 1, 2, 0, 2, 3};
        }
    }

    FluidSurfaceDraw BuildFluidSurfaceDraw(const FluidSimulation2D& fluid)
    {
        FluidSurfaceDraw result;
        result.Fill.DebugName = "Water density surface";
        result.Outline.DebugName = "Water contour is shaded in the field material";
        if (fluid.GetParticles().empty()) return result;
        const auto& settings = fluid.GetSettings();
        const Vector2 extent = settings.BoundsMaximum - settings.BoundsMinimum;
        if (extent.X <= 0.0f || extent.Y <= 0.0f) return result;
        auto field = std::make_shared<Render::ScalarField>();
        // Fixed container-local texels avoid camera jitter and clip splats to rotated walls.
        const float spacing = std::max(0.025f, fluid.GetParticleSpacing() * 0.25f);
        field->Width = std::clamp(static_cast<int>(std::ceil(extent.X / spacing)), 2, 1024);
        field->Height = std::clamp(static_cast<int>(std::ceil(extent.Y / spacing)), 2, 1024);
        field->Material = Render::FieldMaterial::Water;
        field->Rotation = settings.ContainerTransform.Angle;
        field->Values.assign(static_cast<std::size_t>(field->Width * field->Height), 0.0f);
        const Vector2 cell{extent.X / field->Width, extent.Y / field->Height};
        const float radius = std::max(2.0f * settings.ParticleRadius, 1.8f * fluid.GetParticleSpacing());
        const float inverseRadiusSquared = 1.0f / (radius * radius);
        for (const auto& particle : fluid.GetParticles())
        {
            const Vector2 position = settings.ContainerTransform.InverseTransformPoint(particle.Position) - settings.BoundsMinimum;
            const int x0 = std::max(0, static_cast<int>(std::floor((position.X - radius) / cell.X)));
            const int y0 = std::max(0, static_cast<int>(std::floor((position.Y - radius) / cell.Y)));
            const int x1 = std::min(field->Width - 1, static_cast<int>((position.X + radius) / cell.X));
            const int y1 = std::min(field->Height - 1, static_cast<int>((position.Y + radius) / cell.Y));
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x)
                {
                    const Vector2 delta{(x + 0.5f) * cell.X - position.X, (y + 0.5f) * cell.Y - position.Y};
                    const float q = std::max(0.0f, 1.0f - delta.SquareLength() * inverseRadiusSquared);
                    field->Values[y * field->Width + x] += q * q;
                }
        }
        BuildFieldQuad(result.Fill, settings.BoundsMinimum, settings.BoundsMaximum, settings.ContainerTransform);
        result.Fill.Field = std::move(field);
        return result;
    }

    Render::DrawPacket BuildSmokeDraw(const AerodynamicsSimulation2D& air)
    {
        Render::DrawPacket draw;
        draw.DebugName = "Filtered smoke density";
        if (air.GetWidth() == 0) return draw;
        auto field = std::make_shared<Render::ScalarField>();
        field->Width = air.GetWidth(); field->Height = air.GetHeight();
        field->Material = Render::FieldMaterial::Smoke;
        field->Values = air.GetDensityField();
        BuildFieldQuad(draw, air.GetBoundsMinimum(), air.GetBoundsMaximum());
        draw.Field = std::move(field);
        return draw;
    }
}
