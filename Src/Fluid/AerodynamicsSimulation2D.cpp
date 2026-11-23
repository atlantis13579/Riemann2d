#include "AerodynamicsSimulation2D.h"
#include "Cuda/CudaSolvers.h"
#include "Compute/ComputeSolvers.h"

#include "Maths/Maths.h"
#include "Dynamics/PhysicsWorld2D.h"
#include "Collision/Box2.h"
#include "Collision/Sphere2.h"
#include "Collision/Capsule2.h"
#include "Collision/Convex2.h"
#include "Collision/TriangleMesh2.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace Riemann2D
{
    namespace
    {
        bool IsFinite(const Vector2& value)
        {
            return std::isfinite(value.X) && std::isfinite(value.Y);
        }

        struct BilinearSample
        {
            int A, B, C, D;
            float X, Y;
            BilinearSample(int width, int height, float x, float y)
            {
                x = Clamp(x, 0.0f, static_cast<float>(width - 1));
                y = Clamp(y, 0.0f, static_cast<float>(height - 1));
                const int ix = static_cast<int>(x), iy = static_cast<int>(y);
                const int nextX = std::min(ix + 1, width - 1), nextY = std::min(iy + 1, height - 1);
                A = iy * width + ix; B = iy * width + nextX;
                C = nextY * width + ix; D = nextY * width + nextX;
                X = x - static_cast<float>(ix); Y = y - static_cast<float>(iy);
            }
            float Read(const std::vector<float>& field) const
            {
                return (field[A] + X * (field[B] - field[A])) * (1.0f - Y) +
                       (field[C] + X * (field[D] - field[C])) * Y;
            }
            float Limit(float value, const std::vector<float>& field) const
            {
                return Clamp(value, std::min({field[A], field[B], field[C], field[D]}),
                                    std::max({field[A], field[B], field[C], field[D]}));
            }
        };

        template<class Vertices>
        bool InsidePolygon(const Vertices& vertices, Vector2 p, float padding)
        {
            float area = 0;
            for (std::size_t i = 0; i < vertices.size(); ++i)
                area += vertices[i].Cross(vertices[(i + 1) % vertices.size()]);
            const float orientation = area >= 0 ? 1.0f : -1.0f;
            for (std::size_t i = 0; i < vertices.size(); ++i)
            {
                const Vector2 edge = vertices[(i + 1) % vertices.size()] - vertices[i];
                if (orientation * edge.Cross(p - vertices[i]) < -padding * edge.Length()) return false;
            }
            return true;
        }

        bool Contains(const RigidBody2D& body, Vector2 position, float padding)
        {
            const Vector2 p = body.GetTransform().InverseTransformPoint(position);
            const Shape2& shape = body.GetShape();
            switch (shape.GetType())
            {
            case ShapeType2::Sphere2:
            {
                const float r = static_cast<const Sphere2&>(shape).GetRadius() + padding;
                return p.SquareLength() <= r * r;
            }
            case ShapeType2::Box:
                return p.Abs() <= static_cast<const Box2&>(shape).GetHalfExtent() + padding;
            case ShapeType2::Capsule2:
            {
                const auto& capsule = static_cast<const Capsule2&>(shape);
                const Vector2 offset = p - Vector2{Clamp(p.X, -capsule.GetHalfLength(), capsule.GetHalfLength()),0};
                const float r = capsule.GetRadius() + padding;
                return offset.SquareLength() <= r * r;
            }
            case ShapeType2::Convex2:
                return InsidePolygon(static_cast<const Convex2&>(shape).GetVertices(), p, padding);
            case ShapeType2::TriangleMesh2:
            {
                const auto& mesh = static_cast<const TriangleMesh2&>(shape);
                for (std::size_t i = 0; i < mesh.GetTriangleCount(); ++i)
                    if (InsidePolygon(mesh.GetTriangle(i), p, padding)) return true;
                return false;
            }
            }
            return false;
        }
    }

    bool AerodynamicsSimulation2D::Initialize(const Vector2& minimum, const Vector2& maximum,
                                               int width, int height)
    {
        Clear();
        if (width < 3 || height < 3 || width > 4096 || height > 4096 ||
            !std::isfinite(minimum.X + minimum.Y + maximum.X + maximum.Y) ||
            minimum.X >= maximum.X || minimum.Y >= maximum.Y) return false;
        BoundsMinimum = minimum; BoundsMaximum = maximum;
        Width = width; Height = height;
        CellSize = {(maximum.X - minimum.X) / width, (maximum.Y - minimum.Y) / height};
        InverseCellSize = {1.0f / CellSize.X, 1.0f / CellSize.Y};
        const std::size_t count = static_cast<std::size_t>(width * height);
        Density.assign(count, 0.0f); DensityScratch = Density; DensityForward = Density;
        U.assign(static_cast<std::size_t>((width + 1) * height), 0.0f);
        V.assign(static_cast<std::size_t>(width * (height + 1)), 0.0f);
        UScratch = U; VScratch = V;
        CenterVelocity.assign(count, Vector2::Zero());
        Curl.assign(count, 0.0f); VorticityForce.assign(count, Vector2::Zero());
        for (;;)
        {
            PressureLevel level;
            level.Width = width; level.Height = height;
            const float inverseX = width / (maximum.X - minimum.X);
            const float inverseY = height / (maximum.Y - minimum.Y);
            level.WeightX = inverseX * inverseX; level.WeightY = inverseY * inverseY;
            level.Pressure.assign(static_cast<std::size_t>(width * height), 0.0f);
            level.RightHandSide = level.Pressure; level.Residual = level.Pressure;
            level.InverseDiagonal.resize(level.Pressure.size());
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    level.InverseDiagonal[y * width + x] = 1.0f /
                        (((x > 0) + (x + 1 < width)) * level.WeightX +
                         ((y > 0) + (y + 1 < height)) * level.WeightY);
            PressureLevels.push_back(std::move(level));
            if (width <= 4 && height <= 4) break;
            width = (width + 1) / 2; height = (height + 1) / 2;
        }
        return true;
    }

    void AerodynamicsSimulation2D::Clear()
    {
        GpuSolver.Instance.reset();
        Width = Height = 0;
        ProjectionTimeStep = 0.0f;
        RigidWorld = nullptr;
        RigidBoundary = {};
        Density.clear(); DensityScratch.clear(); DensityForward.clear();
        U.clear(); V.clear(); UScratch.clear(); VScratch.clear();
        CenterVelocity.clear(); Curl.clear(); VorticityForce.clear(); PressureLevels.clear();
    }

    Vector2 AerodynamicsSimulation2D::GetCellCenter(int x, int y) const
    {
        return BoundsMinimum + Vector2{(x + 0.5f) * CellSize.X, (y + 0.5f) * CellSize.Y};
    }

    void AerodynamicsSimulation2D::SetRigidBodies(PhysicsWorld2D* world)
    {
        RigidWorld = world;
        SynchronizeRigidBodies();
    }

    void AerodynamicsSimulation2D::SynchronizeRigidBodies()
    {
        RigidBoundary.Update(RigidWorld, BoundsMinimum, CellSize, Width, Height, Density);
        RigidBoundary.EnforceVelocity(U, V);
    }

    void AerodynamicsSimulation2D::AddDensity(const Vector2& position, float amount)
    {
        if (!Width || position.X < BoundsMinimum.X || position.Y < BoundsMinimum.Y ||
            position.X >= BoundsMaximum.X || position.Y >= BoundsMaximum.Y) return;
        const int x = static_cast<int>((position.X - BoundsMinimum.X) * InverseCellSize.X);
        const int y = static_cast<int>((position.Y - BoundsMinimum.Y) * InverseCellSize.Y);
        if (RigidBoundary.IsSolid(x,y)) return;
        Density[y * Width + x] = std::max(0.0f, Density[y * Width + x] + amount);
    }

    void AerodynamicsSimulation2D::AddVelocity(const Vector2& position, const Vector2& velocity)
    {
        if (!Width || position.X < BoundsMinimum.X || position.Y < BoundsMinimum.Y ||
            position.X >= BoundsMaximum.X || position.Y >= BoundsMaximum.Y) return;
        const int x = static_cast<int>((position.X - BoundsMinimum.X) * InverseCellSize.X);
        const int y = static_cast<int>((position.Y - BoundsMinimum.Y) * InverseCellSize.Y);
        if (x > 0) U[y * (Width + 1) + x] += 0.5f * velocity.X;
        if (x + 1 < Width) U[y * (Width + 1) + x + 1] += 0.5f * velocity.X;
        if (y > 0) V[y * Width + x] += 0.5f * velocity.Y;
        if (y + 1 < Height) V[(y + 1) * Width + x] += 0.5f * velocity.Y;
    }

    void AerodynamicsSimulation2D::AddEmitter(const Vector2& position, float radius, float densityRate,
                                               const Vector2& velocity, float deltaTime)
    {
        if (!Width || radius <= 0.0f || deltaTime <= 0.0f) return;
        const int x0 = std::max(0, static_cast<int>(std::floor((position.X - radius - BoundsMinimum.X) * InverseCellSize.X)));
        const int y0 = std::max(0, static_cast<int>(std::floor((position.Y - radius - BoundsMinimum.Y) * InverseCellSize.Y)));
        const int x1 = std::min(Width - 1, static_cast<int>((position.X + radius - BoundsMinimum.X) * InverseCellSize.X));
        const int y1 = std::min(Height - 1, static_cast<int>((position.Y + radius - BoundsMinimum.Y) * InverseCellSize.Y));
        const float blend = 1.0f - std::exp(-12.0f * deltaTime);
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
            {
                if (RigidBoundary.IsSolid(x,y)) continue;
                const float q = std::max(0.0f, 1.0f - (GetCellCenter(x, y) - position).SquareLength() / (radius * radius));
                const float weight = q * q;
                Density[y * Width + x] = std::max(0.0f, Density[y * Width + x] + densityRate * deltaTime * weight);
                const int faceU = y * (Width + 1) + x, faceV = y * Width + x;
                U[faceU] += blend * weight * (velocity.X - U[faceU]);
                U[faceU + 1] += blend * weight * (velocity.X - U[faceU + 1]);
                V[faceV] += blend * weight * (velocity.Y - V[faceV]);
                V[faceV + Width] += blend * weight * (velocity.Y - V[faceV + Width]);
            }
        EnforceBoundary();
    }

    Vector2 AerodynamicsSimulation2D::SampleGridVelocity(float x, float y) const
    {
        return {BilinearSample(Width + 1, Height, x, y - 0.5f).Read(U),
                BilinearSample(Width, Height + 1, x - 0.5f, y).Read(V)};
    }

    Vector2 AerodynamicsSimulation2D::SampleVelocity(const Vector2& position) const
    {
        return Width ? SampleGridVelocity((position.X - BoundsMinimum.X) * InverseCellSize.X,
                                          (position.Y - BoundsMinimum.Y) * InverseCellSize.Y) : Vector2::Zero();
    }

    float AerodynamicsSimulation2D::SampleDensity(const Vector2& position) const
    {
        if (RigidBoundary.IsSolid(static_cast<int>(std::floor((position.X-BoundsMinimum.X)*InverseCellSize.X)),
                                  static_cast<int>(std::floor((position.Y-BoundsMinimum.Y)*InverseCellSize.Y)))) return 0;
        return Width ? BilinearSample(Width, Height, (position.X - BoundsMinimum.X) * InverseCellSize.X - 0.5f,
            (position.Y - BoundsMinimum.Y) * InverseCellSize.Y - 0.5f).Read(Density) : 0.0f;
    }

    float AerodynamicsSimulation2D::GetDensity(int x, int y) const
    {
        return Width ? Density[std::clamp(y, 0, Height - 1) * Width + std::clamp(x, 0, Width - 1)] : 0.0f;
    }

    float AerodynamicsSimulation2D::SamplePressure(const Vector2& position) const
    {
        if (!Width || ProjectionTimeStep <= 0.0f || !std::isfinite(position.X) || !std::isfinite(position.Y) ||
            position.X < BoundsMinimum.X || position.X > BoundsMaximum.X ||
            position.Y < BoundsMinimum.Y || position.Y > BoundsMaximum.Y ||
            !std::isfinite(Settings.AirDensity) || Settings.AirDensity <= 0.0f) return 0.0f;
        // Projection solves u_new = u_star - grad(phi), where phi = dt * p / rho.
        const BilinearSample sample(Width, Height,
            (position.X - BoundsMinimum.X) * InverseCellSize.X - 0.5f,
            (position.Y - BoundsMinimum.Y) * InverseCellSize.Y - 0.5f);
        const auto& p = PressureLevels.front().Pressure;
        const float a = RigidBoundary.ExteriorValue(p,sample.A), b = RigidBoundary.ExteriorValue(p,sample.B);
        const float c = RigidBoundary.ExteriorValue(p,sample.C), d = RigidBoundary.ExteriorValue(p,sample.D);
        return Settings.AirDensity / ProjectionTimeStep *
            ((a + sample.X*(b-a))*(1-sample.Y) + (c + sample.X*(d-c))*sample.Y);
    }

    Vector2 AerodynamicsSimulation2D::GetVelocity(int x, int y) const
    {
        if (!Width) return Vector2::Zero();
        x = std::clamp(x, 0, Width - 1); y = std::clamp(y, 0, Height - 1);
        return {0.5f * (U[y * (Width + 1) + x] + U[y * (Width + 1) + x + 1]),
                0.5f * (V[y * Width + x] + V[(y + 1) * Width + x])};
    }

    void AerodynamicsSimulation2D::EnforceBoundary()
    {
        for (int y = 0; y < Height; ++y) U[y * (Width + 1)] = U[y * (Width + 1) + Width] = 0.0f;
        std::fill(V.begin(), V.begin() + Width, 0.0f);
        std::fill(V.end() - Width, V.end(), 0.0f);
    }

    void AerodynamicsSimulation2D::SmoothPressure(PressureLevel& level, int iterations)
    {
        const int w = level.Width, h = level.Height;
        auto& p = level.Pressure;
        for (int iteration = 0; iteration < iterations; ++iteration)
            for (int parity = 0; parity < 2; ++parity)
            {
#if defined(_OPENMP)
#pragma omp parallel for if (w * h >= 32768) num_threads(4) schedule(static)
#endif
                for (int y = 0; y < h; ++y)
                    for (int x = (y + parity) & 1; x < w; x += 2)
                    {
                        const int i = y * w + x;
                        const float sum = ((x > 0 ? p[i - 1] : 0.0f) + (x + 1 < w ? p[i + 1] : 0.0f)) * level.WeightX +
                            ((y > 0 ? p[i - w] : 0.0f) + (y + 1 < h ? p[i + w] : 0.0f)) * level.WeightY;
                        p[i] += 1.15f * ((sum + level.RightHandSide[i]) * level.InverseDiagonal[i] - p[i]);
                    }
            }
    }

    void AerodynamicsSimulation2D::MultigridCycle(std::size_t index)
    {
        auto& level = PressureLevels[index];
        if (index + 1 == PressureLevels.size()) { SmoothPressure(level, 40); return; }
        SmoothPressure(level, 2);
        const int w = level.Width, h = level.Height;
        const auto& p = level.Pressure;
#if defined(_OPENMP)
#pragma omp parallel for if (w * h >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const int i = y * w + x;
                const float sum = ((x > 0 ? p[i - 1] : 0.0f) + (x + 1 < w ? p[i + 1] : 0.0f)) * level.WeightX +
                    ((y > 0 ? p[i - w] : 0.0f) + (y + 1 < h ? p[i + w] : 0.0f)) * level.WeightY;
                level.Residual[i] = level.RightHandSide[i] - p[i] / level.InverseDiagonal[i] + sum;
            }
        auto& coarse = PressureLevels[index + 1];
        for (int y = 0; y < coarse.Height; ++y)
            for (int x = 0; x < coarse.Width; ++x)
            {
                float sum = 0.0f; int count = 0;
                for (int dy = 0; dy < 2 && 2 * y + dy < h; ++dy)
                    for (int dx = 0; dx < 2 && 2 * x + dx < w; ++dx)
                    { sum += level.Residual[(2 * y + dy) * w + 2 * x + dx]; ++count; }
                coarse.RightHandSide[y * coarse.Width + x] = sum / count;
            }
        // The closed-box Neumann problem has a constant null space.
        const float mean = std::accumulate(coarse.RightHandSide.begin(), coarse.RightHandSide.end(), 0.0f) /
            static_cast<float>(coarse.RightHandSide.size());
        for (auto& value : coarse.RightHandSide) value -= mean;
        std::fill(coarse.Pressure.begin(), coarse.Pressure.end(), 0.0f);
        MultigridCycle(index + 1);
#if defined(_OPENMP)
#pragma omp parallel for if (w * h >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                level.Pressure[y * w + x] += BilinearSample(coarse.Width, coarse.Height,
                    (x + 0.5f) * coarse.Width / w - 0.5f,
                    (y + 0.5f) * coarse.Height / h - 0.5f).Read(coarse.Pressure);
        SmoothPressure(level, 2);
    }

    void AerodynamicsSimulation2D::ProjectVelocityCPU()
    {
        if (!Width) return;
        EnforceBoundary();
        auto& level = PressureLevels.front();
#if defined(_OPENMP)
#pragma omp parallel for if (Width * Height >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 0; y < Height; ++y)
            for (int x = 0; x < Width; ++x)
                level.RightHandSide[y * Width + x] =
                    -(U[y * (Width + 1) + x + 1] - U[y * (Width + 1) + x]) * InverseCellSize.X -
                     (V[(y + 1) * Width + x] - V[y * Width + x]) * InverseCellSize.Y;
        std::fill(level.Pressure.begin(), level.Pressure.end(), 0.0f);
        const int cycles = std::max(1, (Settings.PressureIterations + 7) / 8);
        for (int iteration = 0; iteration < cycles; ++iteration) MultigridCycle(0);
        const auto& pressure = level.Pressure;
#if defined(_OPENMP)
#pragma omp parallel for if (Width * Height >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 0; y < Height; ++y)
            for (int x = 1; x < Width; ++x)
                U[y * (Width + 1) + x] -= (pressure[y * Width + x] - pressure[y * Width + x - 1]) * InverseCellSize.X;
#if defined(_OPENMP)
#pragma omp parallel for if (Width * Height >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 1; y < Height; ++y)
            for (int x = 0; x < Width; ++x)
                V[y * Width + x] -= (pressure[y * Width + x] - pressure[(y - 1) * Width + x]) * InverseCellSize.Y;
    }

    void AerodynamicsSimulation2D::AdvectVelocity(float deltaTime)
    {
        const float dtX = deltaTime * InverseCellSize.X, dtY = deltaTime * InverseCellSize.Y;
        const float ax = std::max(Settings.VelocityDiffusion, 0.0f) * deltaTime * InverseCellSize.X * InverseCellSize.X;
        const float ay = std::max(Settings.VelocityDiffusion, 0.0f) * deltaTime * InverseCellSize.Y * InverseCellSize.Y;
        const float scale = std::min(1.0f, 0.49f / std::max(ax + ay, Epsilon));
        for (int component = 0; component < 2; ++component)
        {
            const int w = Width + (component == 0), h = Height + (component == 1);
            const auto& source = component == 0 ? U : V;
            auto& target = component == 0 ? UScratch : VScratch;
#if defined(_OPENMP)
#pragma omp parallel for if (w * h >= 32768) num_threads(4) schedule(static)
#endif
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                {
                    const float gx = x + (component == 1 ? 0.5f : 0.0f);
                    const float gy = y + (component == 0 ? 0.5f : 0.0f);
                    const Vector2 velocity = SampleGridVelocity(gx, gy);
                    const Vector2 midpoint = SampleGridVelocity(gx - 0.5f * dtX * velocity.X, gy - 0.5f * dtY * velocity.Y);
                    const int i = y * w + x;
                    target[i] = BilinearSample(w, h, x - dtX * midpoint.X, y - dtY * midpoint.Y).Read(source);
                    if (x > 0 && x + 1 < w && y > 0 && y + 1 < h)
                        target[i] += scale * (ax * (source[i - 1] + source[i + 1] - 2.0f * source[i]) +
                                              ay * (source[i - w] + source[i + w] - 2.0f * source[i]));
                }
        }
        U.swap(UScratch); V.swap(VScratch);
        EnforceBoundary();
    }

    void AerodynamicsSimulation2D::ApplyVorticity(float deltaTime)
    {
        if (Settings.VorticityStrength <= 0.0f) return;
#if defined(_OPENMP)
#pragma omp parallel for if (Width * Height >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 0; y < Height; ++y)
            for (int x = 0; x < Width; ++x) CenterVelocity[y * Width + x] = GetVelocity(x, y);
#if defined(_OPENMP)
#pragma omp parallel for if (Width * Height >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 1; y < Height - 1; ++y)
            for (int x = 1; x + 1 < Width; ++x)
            {
                const int i = y * Width + x;
                Curl[i] = 0.5f * ((CenterVelocity[i + 1].Y - CenterVelocity[i - 1].Y) * InverseCellSize.X -
                                  (CenterVelocity[i + Width].X - CenterVelocity[i - Width].X) * InverseCellSize.Y);
            }
#if defined(_OPENMP)
#pragma omp parallel for if (Width * Height >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 1; y < Height - 1; ++y)
            for (int x = 1; x + 1 < Width; ++x)
            {
                const int i = y * Width + x;
                Vector2 gradient{(std::abs(Curl[i + 1]) - std::abs(Curl[i - 1])) * InverseCellSize.X,
                                 (std::abs(Curl[i + Width]) - std::abs(Curl[i - Width])) * InverseCellSize.Y};
                gradient /= std::max(gradient.Length(), 1.0e-5f);
                VorticityForce[i] = (Settings.VorticityStrength * std::min(CellSize.X, CellSize.Y) * Curl[i]) *
                    Vector2{gradient.Y, -gradient.X};
            }
#if defined(_OPENMP)
#pragma omp parallel for if (Width * Height >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 0; y < Height; ++y)
            for (int x = 1; x < Width; ++x)
                U[y * (Width + 1) + x] += 0.5f * deltaTime * (VorticityForce[y * Width + x - 1].X + VorticityForce[y * Width + x].X);
#if defined(_OPENMP)
#pragma omp parallel for if (Width * Height >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 1; y < Height; ++y)
            for (int x = 0; x < Width; ++x)
                V[y * Width + x] += 0.5f * deltaTime * (VorticityForce[(y - 1) * Width + x].Y + VorticityForce[y * Width + x].Y);
    }

    void AerodynamicsSimulation2D::AdvectDensity(float deltaTime)
    {
        const float dtX = deltaTime * InverseCellSize.X, dtY = deltaTime * InverseCellSize.Y;
        const float decay = std::exp(-std::max(Settings.DensityDissipation, 0.0f) * deltaTime);
#if defined(_OPENMP)
#pragma omp parallel for if (Width * Height >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 0; y < Height; ++y)
            for (int x = 0; x < Width; ++x)
            {
                const int i = y * Width + x;
                CenterVelocity[i] = GetVelocity(x, y);
                const Vector2 donor = RigidBoundary.ClipTrace({static_cast<float>(x),static_cast<float>(y)},
                    {x - dtX * CenterVelocity[i].X, y - dtY * CenterVelocity[i].Y});
                DensityForward[i] = RigidBoundary.IsSolid(x,y) ? 0.0f : BilinearSample(Width, Height, donor.X, donor.Y).Read(Density);
            }
#if defined(_OPENMP)
#pragma omp parallel for if (Width * Height >= 32768) num_threads(4) schedule(static)
#endif
        for (int y = 0; y < Height; ++y)
            for (int x = 0; x < Width; ++x)
            {
                const int i = y * Width + x;
                float density = DensityForward[i];
                if (Settings.UseMacCormack)
                {
                    const Vector2 start{static_cast<float>(x),static_cast<float>(y)};
                    const Vector2 backward = RigidBoundary.ClipTrace(start, {x-dtX*CenterVelocity[i].X,y-dtY*CenterVelocity[i].Y});
                    const Vector2 forward = RigidBoundary.ClipTrace(start, {x+dtX*CenterVelocity[i].X,y+dtY*CenterVelocity[i].Y});
                    const float reverse = BilinearSample(Width, Height, forward.X, forward.Y).Read(DensityForward);
                    const BilinearSample donor(Width, Height, backward.X, backward.Y);
                    density = donor.Limit(density + 0.5f * (Density[i] - reverse), Density);
                }
                DensityScratch[i] = decay * std::max(0.0f, density);
            }
        Density.swap(DensityScratch);
        RigidBoundary.RemoveInteriorDensity(Density);
    }

    void AerodynamicsSimulation2D::Step(float deltaTime)
    {
        if (!Width || deltaTime <= 0.0f || !std::isfinite(deltaTime)) return;
        deltaTime = std::min(deltaTime, std::max(Settings.MaximumTimeStep, 1.0e-5f));
        ProjectionTimeStep = deltaTime;
        if (RigidWorld) SynchronizeRigidBodies();
        // GPU flow remains accelerated; shared boundary projection and transport
        // enforce the same impermeability rule on every backend.
        const std::vector<float> previousDensity = RigidBoundary.Active() ? Density : std::vector<float>{};
        if (StepGpu(deltaTime, false))
        {
            if (!RigidBoundary.Active())
            {
                ApplyRigidBodyForces();
                return;
            }
            Density = previousDensity;
        }
        else
        {
            AdvectVelocity(deltaTime);
            ApplyVorticity(deltaTime);
            ProjectVelocityCPU();
        }
        RigidBoundary.Project(U,V,PressureLevels.front().Pressure);
        AdvectDensity(deltaTime);
        ApplyRigidBodyForces();
    }

    bool AerodynamicsSimulation2D::SetBackend(FluidComputeBackend backend)
    {
        BackendError.clear();
        if (backend == FluidComputeBackend::ComputeShader && !GetComputeShaderBackendInfo().Available)
        {
            BackendError = GetComputeShaderBackendInfo().Message;
            return false;
        }
        if (backend == FluidComputeBackend::CUDA && !GetCudaBackendInfo().Available)
        {
            BackendError = GetCudaBackendInfo().Message;
            return false;
        }
        if (Backend != backend) GpuSolver.Instance.reset();
        Backend = backend;
        return true;
    }

    bool AerodynamicsSimulation2D::StepGpu(float deltaTime, bool projectOnly)
    {
        if (Backend == FluidComputeBackend::CPU || !Width) return false;
        if (!GpuSolver.Instance) GpuSolver.Instance = Backend == FluidComputeBackend::CUDA ?
            CreateCudaAirSolver() : CreateComputeAirSolver();
        if (GpuSolver.Instance && GpuSolver.Instance->Step(Settings, CellSize, Width, Height,
            deltaTime, projectOnly, Density, U, V, PressureLevels.front().Pressure, BackendError)) return true;
        if (BackendError.empty()) BackendError = "GPU backend creation failed";
        Backend = FluidComputeBackend::CPU;
        GpuSolver.Instance.reset();
        return false;
    }

    void AerodynamicsSimulation2D::ProjectVelocity(float deltaTime)
    {
        if (!std::isfinite(deltaTime) || deltaTime <= 0.0f) return;
        ProjectionTimeStep = deltaTime;
        if (RigidWorld) SynchronizeRigidBodies();
        if (!StepGpu(0.0f, true)) ProjectVelocityCPU();
        if (Width) RigidBoundary.Project(U,V,PressureLevels.front().Pressure);
    }

    float AerodynamicsSimulation2D::ComputeMeanAbsoluteDivergence() const
    {
        if (!Width) return 0.0f;
        double sum = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for if (Width * Height >= 32768) num_threads(4) schedule(static) reduction(+:sum)
#endif
        for (int y = 0; y < Height; ++y)
            for (int x = 0; x < Width; ++x)
                sum += std::abs((U[y * (Width + 1) + x + 1] - U[y * (Width + 1) + x]) * InverseCellSize.X +
                               (V[(y + 1) * Width + x] - V[y * Width + x]) * InverseCellSize.Y);
        return static_cast<float>(sum / (Width * Height));
    }

    bool AerodynamicsSimulation2D::EvaluateSurfaceForce(const RigidBody2D& body,
                                                        const AerodynamicSurfaceSampler2D& sampler,
                                                        const AerodynamicForceSettings2D& settings,
                                                        AerodynamicForceResult2D& result)
    {
        result = {};
        if (body.GetType() != BodyType2D::Dynamic || !sampler ||
            settings.SurfaceSamples < 8 || settings.SurfaceSamples > 256 ||
            !std::isfinite(settings.DragCoefficient) || settings.DragCoefficient < 0.0f ||
            !std::isfinite(settings.MaximumLinearAcceleration) || settings.MaximumLinearAcceleration <= 0.0f ||
            !std::isfinite(settings.MaximumAngularAcceleration) || settings.MaximumAngularAcceleration <= 0.0f)
            return false;

        AerodynamicForceResult2D accumulated;
        bool valid = true;
        const Transform2& transform = body.GetTransform();
        const auto sample = [&](const Vector2& localPoint, const Vector2& localNormal, float length)
        {
            const Vector2 offset = transform.TransformVector(localPoint);
            const Vector2 normal = transform.TransformVector(localNormal);
            const auto field = sampler(transform.Position + offset);
            if (!std::isfinite(field.Pressure) || !std::isfinite(field.Density) ||
                field.Density < 0.0f || !IsFinite(field.Velocity)) { valid = false; return; }
            const Vector2 relative = field.Velocity - body.GetLinearVelocity() -
                Cross(body.GetAngularVelocity(), offset);
            const float incident = std::max(0.0f, -relative.Dot(normal));
            const Vector2 pressure = -field.Pressure * length * normal;
            const Vector2 drag = -0.5f * settings.DragCoefficient * field.Density *
                incident * incident * length * normal;
            accumulated.PressureForce += pressure;
            accumulated.DragForce += drag;
            accumulated.Torque += offset.Cross(pressure + drag);
            ++accumulated.SampleCount;
        };
        const Shape2& shape = body.GetShape();
        if (shape.GetType() == ShapeType2::Sphere2)
        {
            const float radius = static_cast<const Sphere2&>(shape).GetRadius();
            const float angleStep = 2.0f * Pi / settings.SurfaceSamples;
            for (int i = 0; i < settings.SurfaceSamples; ++i)
            {
                const float angle = (i + 0.5f) * angleStep;
                const Vector2 normal{std::cos(angle), std::sin(angle)};
                sample(radius * normal, normal, radius * angleStep);
            }
        }
        else if (shape.GetType() == ShapeType2::Capsule2)
        {
            const auto& capsule = static_cast<const Capsule2&>(shape);
            const float radius = capsule.GetRadius(), halfLength = capsule.GetHalfLength();
            const int count = std::max(4, settings.SurfaceSamples / 4);
            for (int i = 0; i < count; ++i)
            {
                const float x = -halfLength + (i + 0.5f) * 2.0f * halfLength / count;
                sample({x, radius}, {0, 1}, 2.0f * halfLength / count);
                sample({x, -radius}, {0, -1}, 2.0f * halfLength / count);
                const float angle = -0.5f * Pi + (i + 0.5f) * Pi / count;
                const Vector2 normal{std::cos(angle), std::sin(angle)};
                sample(Vector2{halfLength, 0} + radius * normal, normal, radius * Pi / count);
                sample(Vector2{-halfLength, 0} - radius * normal, -normal, radius * Pi / count);
            }
        }
        else
        {
            std::vector<Vector2> vertices;
            if (shape.GetType() == ShapeType2::Box)
            {
                const Vector2 half = static_cast<const Box2&>(shape).GetHalfExtent();
                vertices = {{-half.X,-half.Y}, {half.X,-half.Y}, {half.X,half.Y}, {-half.X,half.Y}};
            }
            else if (shape.GetType() == ShapeType2::Convex2)
                vertices = static_cast<const Convex2&>(shape).GetVertices();
            else return false;
            float perimeter = 0.0f;
            for (std::size_t i = 0; i < vertices.size(); ++i)
                perimeter += (vertices[(i + 1) % vertices.size()] - vertices[i]).Length();
            for (std::size_t i = 0; i < vertices.size(); ++i)
            {
                const Vector2 edge = vertices[(i + 1) % vertices.size()] - vertices[i];
                const float length = edge.Length();
                const Vector2 normal{edge.Y / length, -edge.X / length};
                const int count = std::max(1, static_cast<int>(std::ceil(settings.SurfaceSamples * length / perimeter)));
                for (int j = 0; j < count; ++j)
                    sample(vertices[i] + ((j + 0.5f) / count) * edge, normal, length / count);
            }
        }
        accumulated.Force = accumulated.PressureForce + accumulated.DragForce;
        if (!valid || !IsFinite(accumulated.Force) || !std::isfinite(accumulated.Torque)) return false;
        const float acceleration = accumulated.Force.Length() * body.GetInverseMass();
        const float angularAcceleration = std::abs(accumulated.Torque) * body.GetInverseInertia();
        if (acceleration > settings.MaximumLinearAcceleration)
            accumulated.ClampScale = settings.MaximumLinearAcceleration / acceleration;
        if (angularAcceleration > settings.MaximumAngularAcceleration)
            accumulated.ClampScale = std::min(accumulated.ClampScale,
                settings.MaximumAngularAcceleration / angularAcceleration);
        accumulated.Force *= accumulated.ClampScale;
        accumulated.PressureForce *= accumulated.ClampScale;
        accumulated.DragForce *= accumulated.ClampScale;
        accumulated.Torque *= accumulated.ClampScale;
        result = accumulated;
        return true;
    }

    bool AerodynamicsSimulation2D::EvaluateBodyForce(const RigidBody2D& body,
                                                     AerodynamicForceResult2D& result) const
    {
        if (!Width) { result = {}; return false; }
        return EvaluateSurfaceForce(body, [this](const Vector2& position)
        {
            if (position.X < BoundsMinimum.X || position.X > BoundsMaximum.X ||
                position.Y < BoundsMinimum.Y || position.Y > BoundsMaximum.Y)
                return AerodynamicSurfaceSample2D{};
            return AerodynamicSurfaceSample2D{SamplePressure(position), Settings.AirDensity,
                                              SampleVelocity(position)};
        }, ForceSettings, result);
    }

    void AerodynamicsSimulation2D::ApplyRigidBodyForces()
    {
        if (!RigidWorld) return;
        for (const auto& body : RigidWorld->GetBodies())
        {
            AerodynamicForceResult2D result;
            if (!EvaluateBodyForce(*body, result)) continue;
            body->ApplyForce(result.Force);
            body->ApplyTorque(result.Torque);
        }
    }

    bool AerodynamicsSimulation2D::RigidBoundaryState::Active() const
    {
        return SolidCount != 0;
    }

    bool AerodynamicsSimulation2D::RigidBoundaryState::IsSolid(int x, int y) const
    {
        return Active() && x >= 0 && x < Width && y >= 0 && y < Height && Solid[y * Width + x] >= 0;
    }

    void AerodynamicsSimulation2D::RigidBoundaryState::Update(const PhysicsWorld2D* world, Vector2 minimum, Vector2 cellSize,
                                   int width, int height, std::vector<float>& density)
    {
        Width = width; Height = height; CellSize = cellSize; SolidCount = 0;
        if (!world || !width || world->GetBodies().empty()) { Solid.clear(); return; }
        const int count = width * height;
        Solid.assign(count, -1); NearestFluid.assign(count, -1);
        BlockU.assign((width + 1) * height, -1); BlockV.assign(width * (height + 1), -1);
        VelocityU.assign(BlockU.size(), 0); VelocityV.assign(BlockV.size(), 0);
        const auto& bodies = world->GetBodies();
        // Reserve a full cell diagonal so interpolated density cannot enter the shape.
        const float padding = cellSize.Length();
        for (std::size_t b = 0; b < bodies.size(); ++b)
        {
            const auto bounds = bodies[b]->GetShape().ComputeAABB(bodies[b]->GetTransform());
            const int x0 = std::max(0, static_cast<int>(std::floor((bounds.Minimum.X-padding-minimum.X)/cellSize.X)));
            const int y0 = std::max(0, static_cast<int>(std::floor((bounds.Minimum.Y-padding-minimum.Y)/cellSize.Y)));
            const int x1 = std::min(width-1, static_cast<int>(std::ceil((bounds.Maximum.X+padding-minimum.X)/cellSize.X)));
            const int y1 = std::min(height-1, static_cast<int>(std::ceil((bounds.Maximum.Y+padding-minimum.Y)/cellSize.Y)));
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x)
                {
                    const Vector2 p = minimum + Vector2{(x + 0.5f) * cellSize.X, (y + 0.5f) * cellSize.Y};
                    if (Contains(*bodies[b], p, padding)) Solid[y * width + x] = static_cast<int>(b);
                }
        }
        std::vector<int> queue;
        for (int i = 0; i < count; ++i)
        {
            if (Solid[i] < 0) { NearestFluid[i] = i; continue; }
            ++SolidCount;
        }
        if (!Active()) return;
        for (int i = 0; i < count; ++i)
        {
            if (Solid[i] < 0) continue;
            const int x = i % width, y = i / width;
            for (const int n : {x > 0 ? i - 1 : i, x + 1 < width ? i + 1 : i,
                               y > 0 ? i - width : i, y + 1 < height ? i + width : i})
                if (Solid[n] < 0) { NearestFluid[i] = n; queue.push_back(i); break; }
        }
        for (std::size_t q = 0; q < queue.size(); ++q)
        {
            const int i = queue[q], x = i % width, y = i / width;
            for (const int n : {x > 0 ? i - 1 : i, x + 1 < width ? i + 1 : i,
                               y > 0 ? i - width : i, y + 1 < height ? i + width : i})
                if (NearestFluid[n] < 0) { NearestFluid[n] = NearestFluid[i]; queue.push_back(n); }
        }
        // Moving solids displace covered smoke into the nearest exterior cell.
        for (int i = 0; i < count; ++i)
            if (Solid[i] >= 0)
            {
                if (NearestFluid[i] >= 0) density[NearestFluid[i]] += density[i];
                density[i] = 0;
            }
        const auto velocity = [&](int b, Vector2 p)
        {
            return bodies[b]->GetLinearVelocity() + Cross(bodies[b]->GetAngularVelocity(),
                p - bodies[b]->GetTransform().Position);
        };
        for (int y = 0; y < height; ++y)
            for (int x = 1; x < width; ++x)
            {
                const int b = std::max(Solid[y * width + x - 1], Solid[y * width + x]);
                if (b < 0) continue;
                const int i = y * (width + 1) + x; BlockU[i] = b;
                VelocityU[i] = velocity(b, minimum + Vector2{x * cellSize.X, (y + 0.5f) * cellSize.Y}).X;
            }
        for (int y = 1; y < height; ++y)
            for (int x = 0; x < width; ++x)
            {
                const int b = std::max(Solid[(y - 1) * width + x], Solid[y * width + x]);
                if (b < 0) continue;
                const int i = y * width + x; BlockV[i] = b;
                VelocityV[i] = velocity(b, minimum + Vector2{(x + 0.5f) * cellSize.X, y * cellSize.Y}).Y;
            }
    }

    void AerodynamicsSimulation2D::RigidBoundaryState::EnforceVelocity(std::vector<float>& u, std::vector<float>& v) const
    {
        if (!Active()) return;
        for (std::size_t i = 0; i < u.size(); ++i) if (BlockU[i] >= 0) u[i] = VelocityU[i];
        for (std::size_t i = 0; i < v.size(); ++i) if (BlockV[i] >= 0) v[i] = VelocityV[i];
    }

    float AerodynamicsSimulation2D::RigidBoundaryState::MaximumBoundaryError(const std::vector<float>& u, const std::vector<float>& v) const
    {
        float error = 0;
        if (!Active()) return error;
        for (std::size_t i = 0; i < u.size(); ++i) if (BlockU[i] >= 0) error = std::max(error, std::abs(u[i] - VelocityU[i]));
        for (std::size_t i = 0; i < v.size(); ++i) if (BlockV[i] >= 0) error = std::max(error, std::abs(v[i] - VelocityV[i]));
        return error;
    }

    void AerodynamicsSimulation2D::RigidBoundaryState::Project(std::vector<float>& u, std::vector<float>& v, std::vector<float>& pressure)
    {
        if (!Active()) return;
        EnforceVelocity(u,v);
        const int count = Width * Height;
        const double wx = 1.0 / (CellSize.X * CellSize.X), wy = 1.0 / (CellSize.Y * CellSize.Y);
        Diagonal.assign(count,0); Correction.assign(count,0); Residual.assign(count,0);
        Direction.assign(count,0); Product.assign(count,0); Preconditioned.assign(count,0);
        RightWeight.assign(count,0); UpWeight.assign(count,0); Factor.assign(count,0); Forward.assign(count,0);
        for (int i = 0; i < count; ++i)
        {
            if (Solid[i] >= 0) continue;
            const int x = i % Width, y = i / Width;
            Diagonal[i] = wx * ((x > 0 && Solid[i-1] < 0) + (x+1 < Width && Solid[i+1] < 0)) +
                wy * ((y > 0 && Solid[i-Width] < 0) + (y+1 < Height && Solid[i+Width] < 0));
            if (Diagonal[i] == 0) continue;
            Residual[i] = -(u[y*(Width+1)+x+1]-u[y*(Width+1)+x])/CellSize.X -
                (v[(y+1)*Width+x]-v[y*Width+x])/CellSize.Y;
            RightWeight[i] = x+1 < Width && Solid[i+1] < 0 ? wx : 0;
            UpWeight[i] = y+1 < Height && Solid[i+Width] < 0 ? wy : 0;
        }
        // MIC(0), following the pressure-solve formulation in Bridson's course notes:
        // https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf
        // Account for discarded fill in the diagonal while keeping a positive factor.
        for (int y = 0; y < Height; ++y)
            for (int x = 0; x < Width; ++x)
            {
                const int i = y*Width+x;
                if (Diagonal[i] == 0) continue;
                double e = Diagonal[i];
                if (x > 0)
                {
                    const double left = RightWeight[i-1], f = Factor[i-1];
                    e -= (left*left + 0.97*left*UpWeight[i-1])*f*f;
                }
                if (y > 0)
                {
                    const double down = UpWeight[i-Width], f = Factor[i-Width];
                    e -= (down*down + 0.97*down*RightWeight[i-Width])*f*f;
                }
                if (e < 0.25*Diagonal[i]) e = Diagonal[i];
                Factor[i] = 1.0/std::sqrt(e);
            }
        const auto precondition = [&]
        {
            for (int y = 0; y < Height; ++y)
                for (int x = 0; x < Width; ++x)
                {
                    const int i = y*Width+x;
                    double value = Residual[i];
                    if (x > 0) value += RightWeight[i-1]*Factor[i-1]*Forward[i-1];
                    if (y > 0) value += UpWeight[i-Width]*Factor[i-Width]*Forward[i-Width];
                    Forward[i] = value*Factor[i];
                }
            for (int y = Height-1; y >= 0; --y)
                for (int x = Width-1; x >= 0; --x)
                {
                    const int i = y*Width+x;
                    double value = Forward[i];
                    if (x+1 < Width) value += RightWeight[i]*Factor[i]*Preconditioned[i+1];
                    if (y+1 < Height) value += UpWeight[i]*Factor[i]*Preconditioned[i+Width];
                    Preconditioned[i] = value*Factor[i];
                }
        };
        // Remove roundoff in the closed-domain Neumann compatibility condition.
        const double mean = std::accumulate(Residual.begin(),Residual.end(),0.0)/std::max(1,count-SolidCount);
        for (int i = 0; i < count; ++i) if (Diagonal[i] > 0) Residual[i] -= mean;
        precondition(); Direction = Preconditioned;
        double rz = std::inner_product(Residual.begin(), Residual.end(), Preconditioned.begin(), 0.0);
        const double tolerance = std::max(1.e-18, rz * 1.e-10);
        for (int iteration = 0; iteration < std::min(1024, 4 * std::max(Width,Height)) && rz > tolerance; ++iteration)
        {
            for (int y = 0; y < Height; ++y)
                for (int x = 0; x < Width; ++x)
                {
                const int i = y*Width+x;
                if (Diagonal[i] == 0) { Product[i] = 0; continue; }
                Product[i] = Diagonal[i]*Direction[i] -
                    (x > 0 ? RightWeight[i-1]*Direction[i-1] : 0) - (x+1 < Width ? RightWeight[i]*Direction[i+1] : 0) -
                    (y > 0 ? UpWeight[i-Width]*Direction[i-Width] : 0) - (y+1 < Height ? UpWeight[i]*Direction[i+Width] : 0);
            }
            const double denominator = std::inner_product(Direction.begin(), Direction.end(), Product.begin(), 0.0);
            if (denominator <= 1.e-30) break;
            const double alpha = rz / denominator;
            for (int i = 0; i < count; ++i)
            {
                Correction[i] += alpha * Direction[i]; Residual[i] -= alpha * Product[i];
            }
            precondition();
            const double next = std::inner_product(Residual.begin(), Residual.end(), Preconditioned.begin(), 0.0);
            const double beta = next / rz;
            for (int i = 0; i < count; ++i) Direction[i] = Preconditioned[i] + beta * Direction[i];
            rz = next;
        }
        for (int y = 0; y < Height; ++y)
            for (int x = 1; x < Width; ++x)
                if (BlockU[y*(Width+1)+x] < 0)
                    u[y*(Width+1)+x] -= static_cast<float>((Correction[y*Width+x]-Correction[y*Width+x-1])/CellSize.X);
        for (int y = 1; y < Height; ++y)
            for (int x = 0; x < Width; ++x)
                if (BlockV[y*Width+x] < 0)
                    v[y*Width+x] -= static_cast<float>((Correction[y*Width+x]-Correction[(y-1)*Width+x])/CellSize.Y);
        for (int i = 0; i < count; ++i) pressure[i] += static_cast<float>(Correction[i]);
        EnforceVelocity(u,v);
    }

    float AerodynamicsSimulation2D::RigidBoundaryState::ExteriorValue(const std::vector<float>& values, int i) const
    {
        return Active() && Solid[i] >= 0 ? (NearestFluid[i] >= 0 ? values[NearestFluid[i]] : 0.0f) : values[i];
    }

    Vector2 AerodynamicsSimulation2D::RigidBoundaryState::ClipTrace(Vector2 start, Vector2 end) const
    {
        if (!Active()) return end;
        const Vector2 delta = end-start;
        const int steps = std::max(1, static_cast<int>(std::ceil(std::max(std::abs(delta.X),std::abs(delta.Y))*3)));
        Vector2 previous = start;
        for (int i = 1; i <= steps; ++i)
        {
            const Vector2 p = start + (static_cast<float>(i)/steps)*delta;
            if (IsSolid(static_cast<int>(std::floor(p.X+0.5f)),static_cast<int>(std::floor(p.Y+0.5f)))) return previous;
            previous = p;
        }
        return end;
    }

    void AerodynamicsSimulation2D::RigidBoundaryState::RemoveInteriorDensity(std::vector<float>& density) const
    {
        if (!Active()) return;
        for (std::size_t i = 0; i < density.size(); ++i) if (Solid[i] >= 0) density[i] = 0;
    }
}
