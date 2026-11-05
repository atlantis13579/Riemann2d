#include "SnowSimulation2D.h"
#include "SnowGpuSolver2D.h"
#include "Fluid/FluidComputeBackend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>

#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define R2_MPM_SSE 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define R2_MPM_NEON 1
#endif

namespace Riemann2D
{
    using Continuum::Matrix2;

    namespace
    {
        Vector2 Apply(const Matrix2& m, Vector2 v) { return {m.X(v.X, v.Y), m.Y(v.X, v.Y)}; }
        Matrix2 Outer(Vector2 a, Vector2 b) { return {a.X * b.X, a.X * b.Y, a.Y * b.X, a.Y * b.Y}; }

        struct Stencil
        {
            int Index[12]{};
            float Weight[12]{}, X[12]{}, Y[12]{}, GradientX[12]{}, GradientY[12]{};
        };

        Stencil MakeStencil(const SnowParticle2D& particle, const SnowSettings2D& settings, int width)
        {
            Stencil result;
            const float inverse = 1 / settings.CellSize;
            const Vector2 grid = (particle.Position - settings.BoundsMinimum) * inverse;
            const int bx = static_cast<int>(std::floor(grid.X - 0.5f)), by = static_cast<int>(std::floor(grid.Y - 0.5f));
            const float fx = grid.X - bx, fy = grid.Y - by;
            for (int y = 0; y < 3; ++y)
                for (int x = 0; x < 3; ++x)
                {
                    const int k = y * 3 + x;
                    const float wx = Continuum::Weight(x, fx), wy = Continuum::Weight(y, fy);
                    result.Index[k] = (by + y) * width + bx + x;
                    result.Weight[k] = wx * wy;
                    result.X[k] = (x - fx) * settings.CellSize;
                    result.Y[k] = (y - fy) * settings.CellSize;
                    if (settings.Transfer == MpmTransfer2D::MLS)
                    {
                        result.GradientX[k] = 4 * inverse * inverse * wx * wy * result.X[k];
                        result.GradientY[k] = 4 * inverse * inverse * wx * wy * result.Y[k];
                    }
                    else
                    {
                        result.GradientX[k] = Continuum::WeightGradient(x, fx) * wy * inverse;
                        result.GradientY[k] = wx * Continuum::WeightGradient(y, fy) * inverse;
                    }
                }
            return result;
        }

        // SIMD processes four stencil nodes together; only the final scatter is scalar.
        void TransferBatch(const SnowParticle2D& p, const Matrix2& affine, const Matrix2& stress,
                           const Stencil& s, int k, float* mass, float* px, float* py, float* fx, float* fy)
        {
#if R2_MPM_SSE
            const __m128 w = _mm_loadu_ps(s.Weight + k), x = _mm_loadu_ps(s.X + k), y = _mm_loadu_ps(s.Y + k);
            const __m128 gx = _mm_loadu_ps(s.GradientX + k), gy = _mm_loadu_ps(s.GradientY + k);
            const __m128 m = _mm_mul_ps(w, _mm_set1_ps(p.Mass));
            _mm_storeu_ps(mass, m);
            _mm_storeu_ps(px, _mm_mul_ps(m, _mm_add_ps(_mm_set1_ps(p.Velocity.X), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(affine.A), x), _mm_mul_ps(_mm_set1_ps(affine.B), y)))));
            _mm_storeu_ps(py, _mm_mul_ps(m, _mm_add_ps(_mm_set1_ps(p.Velocity.Y), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(affine.C), x), _mm_mul_ps(_mm_set1_ps(affine.D), y)))));
            _mm_storeu_ps(fx, _mm_mul_ps(_mm_set1_ps(-p.Volume), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(stress.A), gx), _mm_mul_ps(_mm_set1_ps(stress.B), gy))));
            _mm_storeu_ps(fy, _mm_mul_ps(_mm_set1_ps(-p.Volume), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(stress.C), gx), _mm_mul_ps(_mm_set1_ps(stress.D), gy))));
#elif R2_MPM_NEON
            const float32x4_t w = vld1q_f32(s.Weight + k), x = vld1q_f32(s.X + k), y = vld1q_f32(s.Y + k);
            const float32x4_t gx = vld1q_f32(s.GradientX + k), gy = vld1q_f32(s.GradientY + k);
            const float32x4_t m = vmulq_n_f32(w, p.Mass);
            vst1q_f32(mass, m);
            vst1q_f32(px, vmulq_f32(m, vaddq_f32(vdupq_n_f32(p.Velocity.X), vaddq_f32(vmulq_n_f32(x, affine.A), vmulq_n_f32(y, affine.B)))));
            vst1q_f32(py, vmulq_f32(m, vaddq_f32(vdupq_n_f32(p.Velocity.Y), vaddq_f32(vmulq_n_f32(x, affine.C), vmulq_n_f32(y, affine.D)))));
            vst1q_f32(fx, vmulq_n_f32(vaddq_f32(vmulq_n_f32(gx, stress.A), vmulq_n_f32(gy, stress.B)), -p.Volume));
            vst1q_f32(fy, vmulq_n_f32(vaddq_f32(vmulq_n_f32(gx, stress.C), vmulq_n_f32(gy, stress.D)), -p.Volume));
#else
            for (int lane = 0; lane < 4; ++lane)
            {
                mass[lane] = s.Weight[k + lane] * p.Mass;
                px[lane] = mass[lane] * (p.Velocity.X + affine.X(s.X[k + lane], s.Y[k + lane]));
                py[lane] = mass[lane] * (p.Velocity.Y + affine.Y(s.X[k + lane], s.Y[k + lane]));
                fx[lane] = -p.Volume * stress.X(s.GradientX[k + lane], s.GradientY[k + lane]);
                fy[lane] = -p.Volume * stress.Y(s.GradientX[k + lane], s.GradientY[k + lane]);
            }
#endif
        }
    }

    SnowSimulation2D::SnowSimulation2D(const SnowSettings2D& settings) : Settings(settings) { PrepareGrid(); }
    SnowSimulation2D::~SnowSimulation2D() = default;

    bool SnowSimulation2D::IsSimdAvailable()
    {
#if R2_MPM_SSE || R2_MPM_NEON
        return true;
#else
        return false;
#endif
    }

    bool SnowSimulation2D::SetBackend(ContinuumBackend backend)
    {
        BackendError.clear();
        if (backend < ContinuumBackend::CPU || backend > ContinuumBackend::Metal)
        {
            BackendError = "Unknown deformation backend";
            return false;
        }
        if (backend == ContinuumBackend::SIMD && !IsSimdAvailable())
        {
            BackendError = "SIMD is unavailable on this CPU architecture";
            return false;
        }
        if (backend == ContinuumBackend::CUDA)
        {
            const auto& info = GetCudaBackendInfo();
            if (!info.Available) { BackendError = info.Message; return false; }
            auto solver = CreateCudaSnowSolver2D();
            if (!solver) { BackendError = "CUDA deformation support was not built"; return false; }
            GpuSolver = std::move(solver);
            Backend = backend;
            return true;
        }
        if (backend == ContinuumBackend::ComputeShader || backend == ContinuumBackend::Metal)
        {
            const auto& info = GetComputeShaderBackendInfo();
            if (!info.Available) { BackendError = info.Message; return false; }
            if (backend == ContinuumBackend::Metal && info.API != "Metal")
            {
                BackendError = "Metal requires an Apple Metal device";
                return false;
            }
            GpuSolver = CreateComputeSnowSolver2D();
            Backend = backend;
            return true;
        }
        GpuSolver.reset();
        Backend = backend;
        return true;
    }

    void SnowSimulation2D::ValidateSettings() const
    {
        const auto& m = Settings.Material;
        const Vector2 span = Settings.BoundsMaximum - Settings.BoundsMinimum;
        if (!std::isfinite(Settings.CellSize) || Settings.CellSize <= 0 ||
            !std::isfinite(span.X + span.Y) || span.X < 6 * Settings.CellSize || span.Y < 6 * Settings.CellSize ||
            span.X * span.Y / (Settings.CellSize * Settings.CellSize) > 2000000 ||
            !std::isfinite(Settings.Density) || Settings.Density <= 0 ||
            !std::isfinite(m.YoungModulus) || m.YoungModulus <= 0 ||
            !std::isfinite(m.PoissonRatio) || m.PoissonRatio < 0 || m.PoissonRatio > 0.45f ||
            !std::isfinite(m.SnowFraction) || m.SnowFraction < 0 || m.SnowFraction > 1 ||
            !std::isfinite(m.Viscosity) || m.Viscosity < 0 || m.Viscosity > 1000 ||
            !std::isfinite(m.Hardening) || m.Hardening < 0 || m.Hardening > 30 ||
            !std::isfinite(m.CompressionYield) || m.CompressionYield <= 0 || m.CompressionYield > 0.5f ||
            !std::isfinite(m.StretchYield) || m.StretchYield <= 0 || m.StretchYield > 0.5f ||
            !std::isfinite(m.FrictionAngle) || m.FrictionAngle < 0 || m.FrictionAngle > 60 ||
            !std::isfinite(Settings.Gravity.X + Settings.Gravity.Y) || Settings.MaximumSubsteps < 1 || Settings.MaximumSubsteps > 256 ||
            !std::isfinite(Settings.FrameBudgetMilliseconds) || Settings.FrameBudgetMilliseconds < 0 ||
            !std::isfinite(Settings.FlipRatio) || Settings.FlipRatio < 0 || Settings.FlipRatio > 1 ||
            !std::isfinite(Settings.FixedBedHeight) || Settings.FixedBedHeight >= Settings.BoundsMaximum.Y ||
            Settings.Transfer < MpmTransfer2D::MPM || Settings.Transfer > MpmTransfer2D::MLS)
            throw std::invalid_argument("Invalid MPM material, grid or integration settings");
    }

    void SnowSimulation2D::PrepareGrid()
    {
        ValidateSettings();
        Width = static_cast<int>(std::ceil((Settings.BoundsMaximum.X - Settings.BoundsMinimum.X) / Settings.CellSize)) + 1;
        Height = static_cast<int>(std::ceil((Settings.BoundsMaximum.Y - Settings.BoundsMinimum.Y) / Settings.CellSize)) + 1;
        Grid.resize(static_cast<std::size_t>(Width) * Height);
        const Vector2 low = Settings.BoundsMinimum + 1.5f * Settings.CellSize;
        const Vector2 high = Settings.BoundsMaximum - 1.5f * Settings.CellSize;
        for (SnowParticle2D& p : Particles)
        {
            const Vector2 confined = p.Position.Max(low).Min(high);
            if (confined != p.Position)
            {
                p.Position = confined;
                p.Velocity = {};
                p.Affine = {};
            }
        }
    }

    void SnowSimulation2D::Clear()
    {
        Particles.clear();
        PendingTime = 0;
        Statistics = {};
        GpuSolver.reset();
        FollowSurface = false;
    }

    void SnowSimulation2D::Reset(ContinuumQuality quality)
    {
        Clear();
        Settings.CellSize = quality == ContinuumQuality::Realtime ? 0.18f : 0.1f;
        Settings.BoundsMinimum.Y = -1.5f * Settings.CellSize;
        Settings.MaximumSubsteps = quality == ContinuumQuality::Realtime ? 24 : 64;
        Settings.FrameBudgetMilliseconds = quality == ContinuumQuality::Realtime ? 6.0f : 0;
        // Keep a thin compacted base anchored while the surface remains deformable.
        Settings.FixedBedHeight = 0.3f;
        PrepareGrid();
        Capsule = {};
        Capsule.Center.X = Settings.BoundsMinimum.X + Capsule.Radius + 2 * Settings.CellSize;
        CapsuleTarget = Capsule.Center.X;
        AddRectangle({-6.8f, 0.05f}, {6.8f, 1.8f}, Settings.CellSize * 0.5f);
        FollowSurface = true;
        SurfaceHeight = 1.8f;
        SurfaceSampleX = -1000;
        SurfaceDirection = 0;
        Capsule.Center.Y = SurfaceHeight * 0.8f + Capsule.HalfLength + Capsule.Radius;
        GatherStatistics();
    }

    float SnowSimulation2D::GetScrapeDepth() const
    {
        return std::min(SurfaceHeight * 0.2f, std::max(0.0f, SurfaceHeight - Settings.FixedBedHeight));
    }

    void SnowSimulation2D::UpdateSurfaceTarget()
    {
        if (!FollowSurface || !Capsule.Enabled) return;
        const float travel = CapsuleTarget - Capsule.Center.X;
        if (std::fabs(travel) < 0.02f) return;
        const int direction = travel > 0 ? 1 : -1;
        if (direction == SurfaceDirection && std::fabs(Capsule.Center.X - SurfaceSampleX) < 0.5f * Settings.CellSize) return;
        SurfaceSampleX = Capsule.Center.X;
        SurfaceDirection = direction;
        const float probe = Capsule.Center.X + direction * (Capsule.Radius + 2 * Settings.CellSize);
        std::vector<float> heights;
        for (const SnowParticle2D& p : Particles)
            if (std::fabs(p.Position.X - probe) < Settings.CellSize && std::fabs(p.Velocity.Y) < 1.5f)
                heights.push_back(p.Position.Y);
        if (heights.size() < 4) return;
        // Reject isolated airborne grains. Resample only as the tool advances, so idle contact cannot drill down.
        const size_t index = (heights.size() - 1) * 97 / 100;
        std::nth_element(heights.begin(), heights.begin() + index, heights.end());
        SurfaceHeight = std::max(0.0f, heights[index] + 0.25f * Settings.CellSize);
    }

    void SnowSimulation2D::AddRectangle(Vector2 minimum, Vector2 maximum, float spacing, Vector2 velocity)
    {
        PrepareGrid();
        if (!std::isfinite(spacing) || spacing <= 0 || !std::isfinite(minimum.X + minimum.Y + maximum.X + maximum.Y + velocity.X + velocity.Y))
            throw std::invalid_argument("Invalid MPM particle seeding parameters");
        const Vector2 low = Settings.BoundsMinimum + 2 * Settings.CellSize;
        const Vector2 high = Settings.BoundsMaximum - 2 * Settings.CellSize;
        minimum = minimum.Max(low);
        maximum = maximum.Min(high);
        if (minimum.X >= maximum.X || minimum.Y >= maximum.Y) return;
        if ((maximum.X - minimum.X) * (maximum.Y - minimum.Y) / (spacing * spacing) + Particles.size() > 1000000)
            throw std::invalid_argument("MPM particle count exceeds capacity");
        std::uint32_t seed = 0x9e3779b9u;
        const auto jitter = [&]() {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            return (static_cast<float>(seed & 65535u) / 65535.0f - 0.5f) * 0.3f * spacing;
        };
        for (float y = minimum.Y + 0.5f * spacing; y < maximum.Y; y += spacing)
            for (float x = minimum.X + 0.5f * spacing; x < maximum.X; x += spacing)
            {
                SnowParticle2D p;
                p.Position = {x + jitter(), y + jitter()};
                p.Fixed = p.Position.Y <= Settings.FixedBedHeight ? 1.0f : 0.0f;
                p.Velocity = p.Fixed > 0.5f ? Vector2{} : velocity;
                p.Volume = spacing * spacing;
                p.Mass = Settings.Density * p.Volume;
                Particles.push_back(p);
            }
        GatherStatistics();
    }

    Vector2 SnowSimulation2D::Collide(Vector2 position, Vector2 velocity, float padding) const
    {
        if (!Capsule.Enabled) return velocity;
        const Vector2 closest{Capsule.Center.X, Clamp(position.Y, Capsule.Center.Y - Capsule.HalfLength, Capsule.Center.Y + Capsule.HalfLength)};
        const Vector2 offset = position - closest;
        const float length = offset.Length();
        if (length >= Capsule.Radius + padding) return velocity;
        const Vector2 normal = length > 1.e-6f ? offset / length : Vector2{1, 0};
        const Vector2 relative = velocity - Capsule.Velocity;
        const float incoming = relative.Dot(normal);
        if (incoming >= 0) return velocity;
        const Vector2 tangent = relative - normal * incoming;
        const float friction = std::max(0.0f, 1 - Capsule.Friction * (-incoming) / std::max(tangent.Length(), 1.e-6f));
        return Capsule.Velocity + tangent * friction;
    }

    void SnowSimulation2D::StepCpu(float dt, bool simd)
    {
        std::fill(Grid.begin(), Grid.end(), GridNode{});
        for (const SnowParticle2D& p : Particles)
        {
            const Stencil stencil = MakeStencil(p, Settings, Width);
            const Matrix2 stress = Continuum::KirchhoffStress(p.Elastic, p.Affine, p.PlasticVolume, Settings.Material);
            const Matrix2 affine = Settings.Transfer == MpmTransfer2D::MPM ? Matrix2{} : p.Affine;
            if (simd)
            {
                for (int k = 0; k < 12; k += 4)
                {
                    float mass[4], px[4], py[4], fx[4], fy[4];
                    TransferBatch(p, affine, stress, stencil, k, mass, px, py, fx, fy);
                    for (int lane = 0; lane < 4 && k + lane < 9; ++lane)
                    {
                        GridNode& node = Grid[stencil.Index[k + lane]];
                        node.Mass += mass[lane];
                        node.Momentum += Vector2{px[lane], py[lane]};
                        node.Force += Vector2{fx[lane], fy[lane]};
                    }
                }
            }
            else
            {
                for (int k = 0; k < 9; ++k)
                {
                    GridNode& node = Grid[stencil.Index[k]];
                    const float mass = stencil.Weight[k] * p.Mass;
                    node.Mass += mass;
                    node.Momentum += (p.Velocity + Apply(affine, {stencil.X[k], stencil.Y[k]})) * mass;
                    node.Force -= Apply(stress, {stencil.GradientX[k], stencil.GradientY[k]}) * p.Volume;
                }
            }
        }
        const float dx = Settings.CellSize;
        for (int y = 0; y < Height; ++y)
            for (int x = 0; x < Width; ++x)
            {
                GridNode& node = Grid[y * Width + x];
                if (node.Mass <= 0) continue;
                node.OldVelocity = node.Momentum / node.Mass;
                Vector2 velocity = node.OldVelocity + (node.Force / node.Mass + Settings.Gravity) * dt;
                const Vector2 position = Settings.BoundsMinimum + Vector2{static_cast<float>(x), static_cast<float>(y)} * dx;
                if (x < 2 && velocity.X < 0) velocity.X = 0;
                if (y < 2 && velocity.Y < 0) velocity.Y = 0;
                if (position.X > Settings.BoundsMaximum.X - 2 * dx && velocity.X > 0) velocity.X = 0;
                if (position.Y > Settings.BoundsMaximum.Y - 2 * dx && velocity.Y > 0) velocity.Y = 0;
                node.Momentum = Collide(position, velocity, 0.25f * dx);
                if (position.Y <= Settings.FixedBedHeight) node.Momentum = {};
            }
        const float inverse = 1 / dx;
        for (SnowParticle2D& p : Particles)
        {
            if (p.Fixed > 0.5f) continue;
            const Stencil stencil = MakeStencil(p, Settings, Width);
            Vector2 pic, change;
            Matrix2 gradient, affine;
            for (int k = 0; k < 9; ++k)
            {
                const GridNode& node = Grid[stencil.Index[k]];
                const Vector2 weighted = node.Momentum * stencil.Weight[k];
                pic += weighted;
                change += (node.Momentum - node.OldVelocity) * stencil.Weight[k];
                gradient = gradient + Outer(node.Momentum, {stencil.GradientX[k], stencil.GradientY[k]});
                affine = affine + Outer(weighted, {stencil.X[k], stencil.Y[k]}) * (4 * inverse * inverse);
            }
            p.Velocity = Settings.Transfer == MpmTransfer2D::MPM ? Vector2::Lerp(pic, p.Velocity + change, Settings.FlipRatio) : pic;
            p.Affine = affine;
            p.Elastic = Continuum::ProjectPlasticity((Matrix2::Identity() + gradient * dt) * p.Elastic, p.PlasticVolume, Settings.Material);
            p.Position += pic * dt;
            const Vector2 low = Settings.BoundsMinimum + 1.5f * dx;
            const Vector2 high = Settings.BoundsMaximum - 1.5f * dx;
            if (p.Position.X < low.X || p.Position.X > high.X) p.Velocity.X = 0;
            if (p.Position.Y < low.Y || p.Position.Y > high.Y) p.Velocity.Y = 0;
            p.Position = p.Position.Max(low).Min(high);
            p.Velocity = Collide(p.Position, p.Velocity, 0);
            if (Capsule.Enabled)
            {
                const Vector2 closest{Capsule.Center.X, Clamp(p.Position.Y, Capsule.Center.Y - Capsule.HalfLength, Capsule.Center.Y + Capsule.HalfLength)};
                const Vector2 offset = p.Position - closest;
                const float length = offset.Length();
                if (length < Capsule.Radius)
                    p.Position = closest + (length > 1.e-6f ? offset / length : Vector2{1, 0}) * Capsule.Radius;
            }
            p.Position = p.Position.Max(low).Min(high);
        }
    }

    void SnowSimulation2D::GatherStatistics()
    {
        Statistics.TotalMass = 0;
        Statistics.Momentum = {};
        Statistics.MaximumSpeed = 0;
        for (const SnowParticle2D& p : Particles)
        {
            if (!std::isfinite(p.Position.X + p.Position.Y + p.Velocity.X + p.Velocity.Y + p.Elastic.Determinant() + p.PlasticVolume))
                throw std::runtime_error("MPM produced a non-finite particle state");
            Statistics.TotalMass += p.Mass;
            Statistics.Momentum += p.Velocity * p.Mass;
            Statistics.MaximumSpeed = std::max(Statistics.MaximumSpeed, p.Velocity.Length());
        }
    }

    void SnowSimulation2D::Step(float deltaTime)
    {
        if (!std::isfinite(deltaTime) || deltaTime < 0) throw std::invalid_argument("Invalid MPM time step");
        PrepareGrid();
        if (!std::isfinite(CapsuleTarget + Capsule.Center.X + Capsule.Center.Y + Capsule.Radius + Capsule.HalfLength + Capsule.Friction) ||
            Capsule.Radius <= 0 || Capsule.HalfLength < 0 || Capsule.Friction < 0)
            throw std::invalid_argument("Invalid MPM capsule");
        Statistics.Substeps = 0;
        Statistics.SimulatedTime = 0;
        const auto start = std::chrono::steady_clock::now();
        PendingTime = std::min(PendingTime + deltaTime, 0.25f);
        if (Particles.empty()) PendingTime = 0;
        while (PendingTime > 1.e-7f && Statistics.Substeps < Settings.MaximumSubsteps && !Particles.empty())
        {
            GatherStatistics();
            float mu, lambda;
            Continuum::Lame(Settings.Material, 0.4f, mu, lambda);
            const float wave = std::sqrt((lambda + 2 * mu) / Settings.Density);
            const float cfl = 0.35f * Settings.CellSize / (wave + Statistics.MaximumSpeed + 6.0f);
            const float viscous = 0.15f * Settings.Density * Settings.CellSize * Settings.CellSize / std::max(Settings.Material.Viscosity, 0.001f);
            const float dt = std::min({PendingTime, cfl, viscous, 1.0f / 120});
            const float target = Clamp(CapsuleTarget, Settings.BoundsMinimum.X + Capsule.Radius + 2 * Settings.CellSize,
                Settings.BoundsMaximum.X - Capsule.Radius - 2 * Settings.CellSize);
            UpdateSurfaceTarget();
            const float desiredY = std::max(Settings.FixedBedHeight, SurfaceHeight - GetScrapeDepth())
                + Capsule.HalfLength + Capsule.Radius;
            Capsule.Velocity = {Clamp((target - Capsule.Center.X) / dt, -6.0f, 6.0f),
                FollowSurface ? Clamp((desiredY - Capsule.Center.Y) / dt, -2.0f, 2.0f) : 0};
            Capsule.Center += Capsule.Velocity * dt;
            if (Backend != ContinuumBackend::CPU && Backend != ContinuumBackend::SIMD)
            {
                if (!GpuSolver) GpuSolver = Backend == ContinuumBackend::CUDA ? CreateCudaSnowSolver2D() : CreateComputeSnowSolver2D();
                if (!GpuSolver || !GpuSolver->Step(Settings, Capsule, dt, Particles, BackendError))
                {
                    if (BackendError.empty()) BackendError = "Continuum GPU solver creation failed";
                    Backend = ContinuumBackend::CPU;
                    GpuSolver.reset();
                    StepCpu(dt, false);
                }
            }
            else StepCpu(dt, Backend == ContinuumBackend::SIMD);
            PendingTime -= dt;
            ++Statistics.Substeps;
            Statistics.SimulatedTime += dt;
            if (Settings.FrameBudgetMilliseconds > 0 && std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - start).count() >= Settings.FrameBudgetMilliseconds) break;
        }
        Statistics.PendingTime = PendingTime;
        Statistics.TimeLimited = PendingTime > 1.e-5f;
        GatherStatistics();
    }
}
