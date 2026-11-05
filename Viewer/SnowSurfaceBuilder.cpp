#include "SnowSurfaceBuilder.h"

#include <algorithm>
#include <cmath>

namespace Riemann2D::Viewer
{
    Render::DrawPacket BuildSnowSurfaceDraw(const SnowSimulation2D& simulation, ContinuumQuality quality)
    {
        Render::DrawPacket draw;
        draw.DebugName = "Snow density surface";
        const SnowSettings2D& settings = simulation.GetSettings();
        const Vector2 minimum = settings.BoundsMinimum, maximum = settings.BoundsMaximum;
        const Vector2 extent = maximum - minimum;
        auto field = std::make_shared<Render::ScalarField>();
        const float spacing = settings.CellSize * (quality == ContinuumQuality::Realtime ? 0.4f : 0.25f);
        field->Width = std::clamp(static_cast<int>(std::ceil(extent.X / spacing)), 2, 1024);
        field->Height = std::clamp(static_cast<int>(std::ceil(extent.Y / spacing)), 2, 1024);
        field->Material = Render::FieldMaterial::SnowSand;
        field->Rotation = settings.Material.SnowFraction;
        field->Values.resize(static_cast<std::size_t>(field->Width) * field->Height, 0);
        const Vector2 cell{extent.X / field->Width, extent.Y / field->Height};
        const float radius = settings.CellSize * (quality == ContinuumQuality::Realtime ? 1.0f : 0.85f);
        const float inverseRadiusSquared = 1 / (radius * radius);
        for (const SnowParticle2D& particle : simulation.GetParticles())
        {
            const Vector2 position = particle.Position - minimum;
            const int x0 = std::max(0, static_cast<int>(std::floor((position.X - radius) / cell.X)));
            const int y0 = std::max(0, static_cast<int>(std::floor((position.Y - radius) / cell.Y)));
            const int x1 = std::min(field->Width - 1, static_cast<int>((position.X + radius) / cell.X));
            const int y1 = std::min(field->Height - 1, static_cast<int>((position.Y + radius) / cell.Y));
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x)
                {
                    const Vector2 delta{(x + 0.5f) * cell.X - position.X, (y + 0.5f) * cell.Y - position.Y};
                    const float q = std::max(0.0f, 1 - delta.SquareLength() * inverseRadiusSquared);
                    field->Values[y * field->Width + x] += (3 * particle.Volume * inverseRadiusSquared / Pi) * q * q;
                }
        }
        draw.Vertices = {{minimum, {}, {0,0}}, {{maximum.X,minimum.Y}, {}, {1,0}},
                         {maximum, {}, {1,1}}, {{minimum.X,maximum.Y}, {}, {0,1}}};
        draw.Indices = {0,1,2,0,2,3};
        draw.Field = std::move(field);
        return draw;
    }
}
