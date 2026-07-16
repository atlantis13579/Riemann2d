#include "AerodynamicsSimulation2D.h"

#include "Maths/Maths.h"

#include <algorithm>
#include <cmath>

namespace Riemann2D
{
    bool AerodynamicsSimulation2D::Initialize(const Vector2& minimum, const Vector2& maximum,
                                               int width, int height)
    {
        Clear();
        if (width < 3 || height < 3 || minimum.X >= maximum.X || minimum.Y >= maximum.Y) return false;
        BoundsMinimum = minimum;
        BoundsMaximum = maximum;
        Width = width;
        Height = height;
        CellSize = {(maximum.X - minimum.X) / static_cast<float>(width),
                    (maximum.Y - minimum.Y) / static_cast<float>(height)};
        const std::size_t count = static_cast<std::size_t>(width * height);
        Density.assign(count, 0.0f);
        DensityScratch.assign(count, 0.0f);
        Pressure.assign(count, 0.0f);
        PressureScratch.assign(count, 0.0f);
        Divergence.assign(count, 0.0f);
        Velocity.assign(count, Vector2::Zero());
        VelocityScratch.assign(count, Vector2::Zero());
        return true;
    }

    void AerodynamicsSimulation2D::Clear()
    {
        Width = 0;
        Height = 0;
        Density.clear(); DensityScratch.clear(); Pressure.clear(); PressureScratch.clear();
        Divergence.clear(); Velocity.clear(); VelocityScratch.clear();
    }

    int AerodynamicsSimulation2D::Index(int x, int y) const
    {
        x = std::max(0, std::min(x, Width - 1));
        y = std::max(0, std::min(y, Height - 1));
        return y * Width + x;
    }

    Vector2 AerodynamicsSimulation2D::GetCellCenter(int x, int y) const
    {
        return {BoundsMinimum.X + (static_cast<float>(x) + 0.5f) * CellSize.X,
                BoundsMinimum.Y + (static_cast<float>(y) + 0.5f) * CellSize.Y};
    }

    void AerodynamicsSimulation2D::AddDensity(const Vector2& position, float amount)
    {
        if (Width == 0) return;
        const int x = static_cast<int>((position.X - BoundsMinimum.X) / CellSize.X);
        const int y = static_cast<int>((position.Y - BoundsMinimum.Y) / CellSize.Y);
        if (x >= 0 && x < Width && y >= 0 && y < Height)
            Density[static_cast<std::size_t>(Index(x, y))] =
                std::max(0.0f, Density[static_cast<std::size_t>(Index(x, y))] + amount);
    }

    void AerodynamicsSimulation2D::AddVelocity(const Vector2& position, const Vector2& velocity)
    {
        if (Width == 0) return;
        const int x = static_cast<int>((position.X - BoundsMinimum.X) / CellSize.X);
        const int y = static_cast<int>((position.Y - BoundsMinimum.Y) / CellSize.Y);
        if (x >= 0 && x < Width && y >= 0 && y < Height)
            Velocity[static_cast<std::size_t>(Index(x, y))] += velocity;
    }

    float AerodynamicsSimulation2D::SampleScalar(const std::vector<float>& field,
                                                  const Vector2& position) const
    {
        const float gridX = (position.X - BoundsMinimum.X) / CellSize.X - 0.5f;
        const float gridY = (position.Y - BoundsMinimum.Y) / CellSize.Y - 0.5f;
        const int x0 = static_cast<int>(std::floor(gridX));
        const int y0 = static_cast<int>(std::floor(gridY));
        const float tx = Clamp(gridX - static_cast<float>(x0), 0.0f, 1.0f);
        const float ty = Clamp(gridY - static_cast<float>(y0), 0.0f, 1.0f);
        const float a = field[static_cast<std::size_t>(Index(x0, y0))] * (1.0f - tx) +
                        field[static_cast<std::size_t>(Index(x0 + 1, y0))] * tx;
        const float b = field[static_cast<std::size_t>(Index(x0, y0 + 1))] * (1.0f - tx) +
                        field[static_cast<std::size_t>(Index(x0 + 1, y0 + 1))] * tx;
        return a * (1.0f - ty) + b * ty;
    }

    Vector2 AerodynamicsSimulation2D::SampleVector(const std::vector<Vector2>& field,
                                                    const Vector2& position) const
    {
        const float gridX = (position.X - BoundsMinimum.X) / CellSize.X - 0.5f;
        const float gridY = (position.Y - BoundsMinimum.Y) / CellSize.Y - 0.5f;
        const int x0 = static_cast<int>(std::floor(gridX));
        const int y0 = static_cast<int>(std::floor(gridY));
        const float tx = Clamp(gridX - static_cast<float>(x0), 0.0f, 1.0f);
        const float ty = Clamp(gridY - static_cast<float>(y0), 0.0f, 1.0f);
        const Vector2 a = field[static_cast<std::size_t>(Index(x0, y0))] * (1.0f - tx) +
                          field[static_cast<std::size_t>(Index(x0 + 1, y0))] * tx;
        const Vector2 b = field[static_cast<std::size_t>(Index(x0, y0 + 1))] * (1.0f - tx) +
                          field[static_cast<std::size_t>(Index(x0 + 1, y0 + 1))] * tx;
        return a * (1.0f - ty) + b * ty;
    }

    float AerodynamicsSimulation2D::SampleDensity(const Vector2& position) const
    {
        return Width > 0 ? SampleScalar(Density, position) : 0.0f;
    }

    Vector2 AerodynamicsSimulation2D::SampleVelocity(const Vector2& position) const
    {
        return Width > 0 ? SampleVector(Velocity, position) : Vector2::Zero();
    }

    float AerodynamicsSimulation2D::GetDensity(int x, int y) const
    {
        return Width > 0 ? Density[static_cast<std::size_t>(Index(x, y))] : 0.0f;
    }

    Vector2 AerodynamicsSimulation2D::GetVelocity(int x, int y) const
    {
        return Width > 0 ? Velocity[static_cast<std::size_t>(Index(x, y))] : Vector2::Zero();
    }

    void AerodynamicsSimulation2D::Step(float deltaTime)
    {
        if (Width == 0 || deltaTime <= 0.0f) return;
        deltaTime = std::min(deltaTime, Settings.MaximumTimeStep);
        DiffuseVelocity(deltaTime);
        ProjectVelocity();
        AdvectVelocity(deltaTime);
        ProjectVelocity();
        AdvectDensity(deltaTime);
    }

    void AerodynamicsSimulation2D::DiffuseVelocity(float deltaTime)
    {
        const float blend = Clamp(Settings.VelocityDiffusion * deltaTime /
            std::max(CellSize.X * CellSize.Y, Epsilon), 0.0f, 0.25f);
        VelocityScratch = Velocity;
        for (int y = 1; y < Height - 1; ++y)
        {
            for (int x = 1; x < Width - 1; ++x)
            {
                const Vector2 average = 0.25f * (Velocity[static_cast<std::size_t>(Index(x - 1, y))] +
                    Velocity[static_cast<std::size_t>(Index(x + 1, y))] +
                    Velocity[static_cast<std::size_t>(Index(x, y - 1))] +
                    Velocity[static_cast<std::size_t>(Index(x, y + 1))]);
                const std::size_t index = static_cast<std::size_t>(Index(x, y));
                VelocityScratch[index] = (1.0f - blend) * Velocity[index] + blend * average;
            }
        }
        Velocity.swap(VelocityScratch);
        EnforceBoundary();
    }

    void AerodynamicsSimulation2D::ProjectVelocity()
    {
        std::fill(Pressure.begin(), Pressure.end(), 0.0f);
        for (int y = 1; y < Height - 1; ++y)
        {
            for (int x = 1; x < Width - 1; ++x)
            {
                Divergence[static_cast<std::size_t>(Index(x, y))] =
                    0.5f * ((Velocity[static_cast<std::size_t>(Index(x + 1, y))].X -
                             Velocity[static_cast<std::size_t>(Index(x - 1, y))].X) / CellSize.X +
                            (Velocity[static_cast<std::size_t>(Index(x, y + 1))].Y -
                             Velocity[static_cast<std::size_t>(Index(x, y - 1))].Y) / CellSize.Y);
            }
        }
        const float dx2 = CellSize.X * CellSize.X;
        const float dy2 = CellSize.Y * CellSize.Y;
        const float denominator = 2.0f * (dx2 + dy2);
        for (int iteration = 0; iteration < std::max(Settings.PressureIterations, 1); ++iteration)
        {
            PressureScratch = Pressure;
            for (int y = 1; y < Height - 1; ++y)
            {
                for (int x = 1; x < Width - 1; ++x)
                {
                    PressureScratch[static_cast<std::size_t>(Index(x, y))] =
                        ((Pressure[static_cast<std::size_t>(Index(x - 1, y))] +
                          Pressure[static_cast<std::size_t>(Index(x + 1, y))]) * dy2 +
                         (Pressure[static_cast<std::size_t>(Index(x, y - 1))] +
                          Pressure[static_cast<std::size_t>(Index(x, y + 1))]) * dx2 -
                         Divergence[static_cast<std::size_t>(Index(x, y))] * dx2 * dy2) / denominator;
                }
            }
            Pressure.swap(PressureScratch);
        }
        for (int y = 1; y < Height - 1; ++y)
        {
            for (int x = 1; x < Width - 1; ++x)
            {
                Vector2& velocity = Velocity[static_cast<std::size_t>(Index(x, y))];
                velocity.X -= 0.5f * (Pressure[static_cast<std::size_t>(Index(x + 1, y))] -
                                      Pressure[static_cast<std::size_t>(Index(x - 1, y))]) / CellSize.X;
                velocity.Y -= 0.5f * (Pressure[static_cast<std::size_t>(Index(x, y + 1))] -
                                      Pressure[static_cast<std::size_t>(Index(x, y - 1))]) / CellSize.Y;
            }
        }
        EnforceBoundary();
    }

    void AerodynamicsSimulation2D::AdvectVelocity(float deltaTime)
    {
        VelocityScratch = Velocity;
        for (int y = 1; y < Height - 1; ++y)
            for (int x = 1; x < Width - 1; ++x)
            {
                const Vector2 center = GetCellCenter(x, y);
                VelocityScratch[static_cast<std::size_t>(Index(x, y))] =
                    SampleVector(Velocity, center - deltaTime * Velocity[static_cast<std::size_t>(Index(x, y))]);
            }
        Velocity.swap(VelocityScratch);
        EnforceBoundary();
    }

    void AerodynamicsSimulation2D::AdvectDensity(float deltaTime)
    {
        const float dissipation = std::exp(-std::max(Settings.DensityDissipation, 0.0f) * deltaTime);
        for (int y = 0; y < Height; ++y)
            for (int x = 0; x < Width; ++x)
            {
                const Vector2 center = GetCellCenter(x, y);
                DensityScratch[static_cast<std::size_t>(Index(x, y))] = dissipation *
                    SampleScalar(Density, center - deltaTime * Velocity[static_cast<std::size_t>(Index(x, y))]);
            }
        Density.swap(DensityScratch);
    }

    void AerodynamicsSimulation2D::EnforceBoundary()
    {
        for (int x = 0; x < Width; ++x)
        {
            Velocity[static_cast<std::size_t>(Index(x, 0))].Y = 0.0f;
            Velocity[static_cast<std::size_t>(Index(x, Height - 1))].Y = 0.0f;
        }
        for (int y = 0; y < Height; ++y)
        {
            Velocity[static_cast<std::size_t>(Index(0, y))].X = 0.0f;
            Velocity[static_cast<std::size_t>(Index(Width - 1, y))].X = 0.0f;
        }
    }

    float AerodynamicsSimulation2D::ComputeMeanAbsoluteDivergence() const
    {
        if (Width < 3 || Height < 3) return 0.0f;
        float sum = 0.0f;
        int count = 0;
        for (int y = 1; y < Height - 1; ++y)
            for (int x = 1; x < Width - 1; ++x)
            {
                const float divergence = 0.5f *
                    ((GetVelocity(x + 1, y).X - GetVelocity(x - 1, y).X) / CellSize.X +
                     (GetVelocity(x, y + 1).Y - GetVelocity(x, y - 1).Y) / CellSize.Y);
                sum += std::abs(divergence);
                ++count;
            }
        return sum / static_cast<float>(count);
    }
}
