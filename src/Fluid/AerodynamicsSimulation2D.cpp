#include "AerodynamicsSimulation2D.h"
#include "Cuda/CudaSolvers.h"
#include "Compute/ComputeSolvers.h"

#include "Maths/Maths.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace Riemann2D
{
    namespace
    {
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
        Density.clear(); DensityScratch.clear(); DensityForward.clear();
        U.clear(); V.clear(); UScratch.clear(); VScratch.clear();
        CenterVelocity.clear(); Curl.clear(); VorticityForce.clear(); PressureLevels.clear();
    }

    Vector2 AerodynamicsSimulation2D::GetCellCenter(int x, int y) const
    {
        return BoundsMinimum + Vector2{(x + 0.5f) * CellSize.X, (y + 0.5f) * CellSize.Y};
    }

    void AerodynamicsSimulation2D::AddDensity(const Vector2& position, float amount)
    {
        if (!Width || position.X < BoundsMinimum.X || position.Y < BoundsMinimum.Y ||
            position.X >= BoundsMaximum.X || position.Y >= BoundsMaximum.Y) return;
        const int x = static_cast<int>((position.X - BoundsMinimum.X) * InverseCellSize.X);
        const int y = static_cast<int>((position.Y - BoundsMinimum.Y) * InverseCellSize.Y);
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
        return Width ? BilinearSample(Width, Height, (position.X - BoundsMinimum.X) * InverseCellSize.X - 0.5f,
            (position.Y - BoundsMinimum.Y) * InverseCellSize.Y - 0.5f).Read(Density) : 0.0f;
    }

    float AerodynamicsSimulation2D::GetDensity(int x, int y) const
    {
        return Width ? Density[std::clamp(y, 0, Height - 1) * Width + std::clamp(x, 0, Width - 1)] : 0.0f;
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
                DensityForward[i] = BilinearSample(Width, Height, x - dtX * CenterVelocity[i].X,
                    y - dtY * CenterVelocity[i].Y).Read(Density);
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
                    const float reverse = BilinearSample(Width, Height, x + dtX * CenterVelocity[i].X,
                        y + dtY * CenterVelocity[i].Y).Read(DensityForward);
                    const BilinearSample donor(Width, Height, x - dtX * CenterVelocity[i].X, y - dtY * CenterVelocity[i].Y);
                    density = donor.Limit(density + 0.5f * (Density[i] - reverse), Density);
                }
                DensityScratch[i] = decay * std::max(0.0f, density);
            }
        Density.swap(DensityScratch);
    }

    void AerodynamicsSimulation2D::Step(float deltaTime)
    {
        if (!Width || deltaTime <= 0.0f || !std::isfinite(deltaTime)) return;
        deltaTime = std::min(deltaTime, std::max(Settings.MaximumTimeStep, 1.0e-5f));
        if (StepGpu(deltaTime, false)) return;
        AdvectVelocity(deltaTime);
        ApplyVorticity(deltaTime);
        ProjectVelocityCPU();
        AdvectDensity(deltaTime);
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
            deltaTime, projectOnly, Density, U, V, BackendError)) return true;
        if (BackendError.empty()) BackendError = "GPU backend creation failed";
        Backend = FluidComputeBackend::CPU;
        GpuSolver.Instance.reset();
        return false;
    }

    void AerodynamicsSimulation2D::ProjectVelocity()
    {
        if (!StepGpu(0.0f, true)) ProjectVelocityCPU();
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
}
