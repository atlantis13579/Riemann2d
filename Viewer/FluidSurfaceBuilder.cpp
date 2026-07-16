#include "FluidSurfaceBuilder.h"

#include "GeometryBuilder.h"
#include "Maths/Maths.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace Riemann2D::Viewer
{
    namespace
    {
        constexpr float SurfaceLevel = 0.5f;
        constexpr int MinimumSurfaceNeighbors = 2;

        Render::Color GetWaterColor(const Vector2& position, float minimumY, float maximumY)
        {
            const float range = std::max(maximumY - minimumY, Epsilon);
            const float height = Clamp((position.Y - minimumY) / range, 0.0f, 1.0f);
            const Render::Color deep{0.025f, 0.19f, 0.48f, 0.94f};
            const Render::Color shallow{0.06f, 0.48f, 0.92f, 0.90f};
            return {
                deep.R + height * (shallow.R - deep.R),
                deep.G + height * (shallow.G - deep.G),
                deep.B + height * (shallow.B - deep.B),
                deep.A + height * (shallow.A - deep.A)
            };
        }

        void AddWaterTriangle(Render::DrawPacket& draw, const Vector2& first,
                              const Vector2& second, const Vector2& third,
                              float minimumY, float maximumY)
        {
            const std::uint32_t base = static_cast<std::uint32_t>(draw.Vertices.size());
            draw.Vertices.push_back({first, GetWaterColor(first, minimumY, maximumY)});
            draw.Vertices.push_back({second, GetWaterColor(second, minimumY, maximumY)});
            draw.Vertices.push_back({third, GetWaterColor(third, minimumY, maximumY)});
            draw.Indices.insert(draw.Indices.end(), {base, base + 1, base + 2});
        }

        void AddWaterPolygon(Render::DrawPacket& draw, const std::vector<Vector2>& polygon,
                             float minimumY, float maximumY)
        {
            for (std::size_t index = 1; index + 1 < polygon.size(); ++index)
            {
                AddWaterTriangle(draw, polygon[0], polygon[index], polygon[index + 1],
                                 minimumY, maximumY);
            }
        }

        Vector2 InterpolateSurface(const Vector2& first, const Vector2& second,
                                   float firstValue, float secondValue)
        {
            const float difference = secondValue - firstValue;
            const float fraction = std::abs(difference) > Epsilon
                ? Clamp((SurfaceLevel - firstValue) / difference, 0.0f, 1.0f)
                : 0.5f;
            return Vector2::Lerp(first, second, fraction);
        }

        void AddSurfaceLine(Render::DrawPacket& draw, const Vector2& first,
                            const Vector2& second, const Vector2& fieldGradient)
        {
            const float gradientLength = fieldGradient.Length();
            const float upwardFacing = gradientLength > Epsilon
                ? Clamp(-fieldGradient.Y / gradientLength, 0.0f, 1.0f)
                : 0.0f;
            const Render::Color side{0.12f, 0.52f, 0.82f, 0.78f};
            const Render::Color crest{0.58f, 0.90f, 1.0f, 1.0f};
            const Render::Color color{
                side.R + upwardFacing * (crest.R - side.R),
                side.G + upwardFacing * (crest.G - side.G),
                side.B + upwardFacing * (crest.B - side.B),
                side.A + upwardFacing * (crest.A - side.A)
            };
            AddLine(draw, first, second, color);
        }

        void AddCellSurface(Render::DrawPacket& fill, Render::DrawPacket& outline,
                            const std::array<Vector2, 4>& corners,
                            const std::array<float, 4>& values,
                            float minimumY, float maximumY, float gridSpacing)
        {
            int mask = 0;
            for (int corner = 0; corner < 4; ++corner)
            {
                if (values[static_cast<std::size_t>(corner)] >= SurfaceLevel)
                {
                    mask |= 1 << corner;
                }
            }
            if (mask == 0)
            {
                return;
            }

            std::array<Vector2, 4> crossings{};
            std::array<bool, 4> hasCrossing{};
            for (int edge = 0; edge < 4; ++edge)
            {
                const int next = (edge + 1) % 4;
                const bool firstInside = (mask & (1 << edge)) != 0;
                const bool secondInside = (mask & (1 << next)) != 0;
                if (firstInside != secondInside)
                {
                    crossings[static_cast<std::size_t>(edge)] = InterpolateSurface(
                        corners[static_cast<std::size_t>(edge)],
                        corners[static_cast<std::size_t>(next)],
                        values[static_cast<std::size_t>(edge)],
                        values[static_cast<std::size_t>(next)]);
                    hasCrossing[static_cast<std::size_t>(edge)] = true;
                }
            }

            const float centerValue = 0.25f * (values[0] + values[1] + values[2] + values[3]);
            const bool ambiguous = mask == 5 || mask == 10;
            if (ambiguous && centerValue < SurfaceLevel)
            {
                if (mask == 5)
                {
                    AddWaterPolygon(fill, {corners[0], crossings[0], crossings[3]},
                                    minimumY, maximumY);
                    AddWaterPolygon(fill, {corners[2], crossings[2], crossings[1]},
                                    minimumY, maximumY);
                }
                else
                {
                    AddWaterPolygon(fill, {corners[1], crossings[1], crossings[0]},
                                    minimumY, maximumY);
                    AddWaterPolygon(fill, {corners[3], crossings[3], crossings[2]},
                                    minimumY, maximumY);
                }
            }
            else
            {
                std::vector<Vector2> polygon;
                polygon.reserve(8);
                for (int edge = 0; edge < 4; ++edge)
                {
                    if ((mask & (1 << edge)) != 0)
                    {
                        polygon.push_back(corners[static_cast<std::size_t>(edge)]);
                    }
                    if (hasCrossing[static_cast<std::size_t>(edge)])
                    {
                        polygon.push_back(crossings[static_cast<std::size_t>(edge)]);
                    }
                }
                AddWaterPolygon(fill, polygon, minimumY, maximumY);
            }

            const Vector2 gradient{
                ((values[1] + values[2]) - (values[0] + values[3])) /
                    (2.0f * gridSpacing),
                ((values[2] + values[3]) - (values[0] + values[1])) /
                    (2.0f * gridSpacing)
            };
            if (ambiguous)
            {
                const bool connectAroundInside = centerValue < SurfaceLevel;
                if ((mask == 5) == connectAroundInside)
                {
                    AddSurfaceLine(outline, crossings[3], crossings[0], gradient);
                    AddSurfaceLine(outline, crossings[1], crossings[2], gradient);
                }
                else
                {
                    AddSurfaceLine(outline, crossings[0], crossings[1], gradient);
                    AddSurfaceLine(outline, crossings[2], crossings[3], gradient);
                }
            }
            else
            {
                std::array<Vector2, 2> segment{};
                int crossingCount = 0;
                for (int edge = 0; edge < 4; ++edge)
                {
                    if (hasCrossing[static_cast<std::size_t>(edge)] && crossingCount < 2)
                    {
                        segment[static_cast<std::size_t>(crossingCount++)] =
                            crossings[static_cast<std::size_t>(edge)];
                    }
                }
                if (crossingCount == 2)
                {
                    AddSurfaceLine(outline, segment[0], segment[1], gradient);
                }
            }
        }
    }

    FluidSurfaceDraw BuildFluidSurfaceDraw(const FluidSimulation2D& fluid)
    {
        FluidSurfaceDraw result;
        result.Fill.DebugName = "Fluid surface fill";
        result.Fill.Topology = Render::PrimitiveTopology::Triangles;
        result.Outline.DebugName = "Fluid surface outline";
        result.Outline.Topology = Render::PrimitiveTopology::Lines;

        const std::vector<FluidParticle2D>& particles = fluid.GetParticles();
        if (particles.empty())
        {
            return result;
        }

        const FluidSettings2D& settings = fluid.GetSettings();
        const float particleRadius = std::max(settings.ParticleRadius, Epsilon);
        const float kernelRadius = std::max(settings.SmoothingRadius * 1.5f,
                                            particleRadius * 4.0f);
        const float kernelRadiusSquared = kernelRadius * kernelRadius;
        const float gridSpacing = std::max(0.04f,
            std::min(settings.SmoothingRadius * 0.32f, particleRadius * 1.75f));

        std::vector<bool> surfaceParticles(particles.size(), false);
        for (std::size_t first = 0; first < particles.size(); ++first)
        {
            int neighborCount = 0;
            for (std::size_t second = 0; second < particles.size(); ++second)
            {
                if (first == second)
                {
                    continue;
                }
                if ((particles[first].Position - particles[second].Position).SquareLength() <=
                    kernelRadiusSquared)
                {
                    ++neighborCount;
                }
            }
            surfaceParticles[first] = neighborCount >= MinimumSurfaceNeighbors;
        }

        Vector2 minimum = Vector2::InfMax();
        Vector2 maximum = Vector2::InfMin();
        bool hasSurfaceParticles = false;
        for (std::size_t index = 0; index < particles.size(); ++index)
        {
            if (!surfaceParticles[index])
            {
                continue;
            }
            minimum = minimum.Min(particles[index].Position);
            maximum = maximum.Max(particles[index].Position);
            hasSurfaceParticles = true;
        }
        if (!hasSurfaceParticles)
        {
            return result;
        }

        minimum -= kernelRadius + gridSpacing;
        maximum += kernelRadius + gridSpacing;
        minimum.X = std::floor(minimum.X / gridSpacing) * gridSpacing;
        minimum.Y = std::floor(minimum.Y / gridSpacing) * gridSpacing;
        maximum.X = std::ceil(maximum.X / gridSpacing) * gridSpacing;
        maximum.Y = std::ceil(maximum.Y / gridSpacing) * gridSpacing;

        const int width = std::max(2, static_cast<int>(
            std::ceil((maximum.X - minimum.X) / gridSpacing)) + 1);
        const int height = std::max(2, static_cast<int>(
            std::ceil((maximum.Y - minimum.Y) / gridSpacing)) + 1);
        std::vector<float> field(static_cast<std::size_t>(width * height), 0.0f);
        const auto fieldIndex = [width](int x, int y)
        {
            return static_cast<std::size_t>(y * width + x);
        };

        for (std::size_t particleIndex = 0; particleIndex < particles.size(); ++particleIndex)
        {
            if (!surfaceParticles[particleIndex])
            {
                continue;
            }
            const Vector2 position = particles[particleIndex].Position;
            const int startX = std::max(0, static_cast<int>(
                std::floor((position.X - kernelRadius - minimum.X) / gridSpacing)));
            const int endX = std::min(width - 1, static_cast<int>(
                std::ceil((position.X + kernelRadius - minimum.X) / gridSpacing)));
            const int startY = std::max(0, static_cast<int>(
                std::floor((position.Y - kernelRadius - minimum.Y) / gridSpacing)));
            const int endY = std::min(height - 1, static_cast<int>(
                std::ceil((position.Y + kernelRadius - minimum.Y) / gridSpacing)));
            for (int y = startY; y <= endY; ++y)
            {
                for (int x = startX; x <= endX; ++x)
                {
                    const Vector2 node = minimum + gridSpacing * Vector2{
                        static_cast<float>(x), static_cast<float>(y)};
                    const float distanceSquared = (node - position).SquareLength();
                    if (distanceSquared >= kernelRadiusSquared)
                    {
                        continue;
                    }
                    const float normalized = 1.0f - distanceSquared / kernelRadiusSquared;
                    field[fieldIndex(x, y)] += normalized * normalized;
                }
            }
        }

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const Vector2 node = minimum + gridSpacing * Vector2{
                    static_cast<float>(x), static_cast<float>(y)};
                const Vector2 localNode =
                    settings.ContainerTransform.InverseTransformPoint(node);
                if (localNode.X < settings.BoundsMinimum.X ||
                    localNode.X > settings.BoundsMaximum.X ||
                    localNode.Y < settings.BoundsMinimum.Y ||
                    localNode.Y > settings.BoundsMaximum.Y)
                {
                    field[fieldIndex(x, y)] = 0.0f;
                }
            }
        }

        for (int y = 0; y + 1 < height; ++y)
        {
            for (int x = 0; x + 1 < width; ++x)
            {
                const Vector2 bottomLeft = minimum + gridSpacing * Vector2{
                    static_cast<float>(x), static_cast<float>(y)};
                const std::array<Vector2, 4> corners{
                    bottomLeft,
                    bottomLeft + Vector2{gridSpacing, 0.0f},
                    bottomLeft + Vector2{gridSpacing, gridSpacing},
                    bottomLeft + Vector2{0.0f, gridSpacing}
                };
                const std::array<float, 4> values{
                    field[fieldIndex(x, y)], field[fieldIndex(x + 1, y)],
                    field[fieldIndex(x + 1, y + 1)], field[fieldIndex(x, y + 1)]
                };
                AddCellSurface(result.Fill, result.Outline, corners, values,
                               minimum.Y, maximum.Y, gridSpacing);
            }
        }
        return result;
    }
}
