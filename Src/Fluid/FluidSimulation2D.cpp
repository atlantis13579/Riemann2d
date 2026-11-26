#include "FluidSimulation2D.h"
#include "Cuda/CudaSolvers.h"


#include "Maths/Maths.h"

#include <algorithm>
#include <cmath>

namespace Riemann2D
{
    void FluidSimulation2D::CreateBlock(const Vector2& minimum, const Vector2& maximum, float spacing)
    {
        Clear();
        spacing = std::max(spacing, 2.0f * Settings.ParticleRadius);
        InitialSpacing = spacing;
        for (float y = minimum.Y; y <= maximum.Y + Epsilon; y += spacing)
        {
            for (float x = minimum.X; x <= maximum.X + Epsilon; x += spacing)
            {
                FluidParticle2D particle;
                particle.Position = Settings.ContainerTransform.TransformPoint({x, y});
                Particles.push_back(particle);
            }
        }
        PreviousPositions.resize(Particles.size());
        VelocityDeltas.resize(Particles.size());
        UpdateDensities();
    }

    void FluidSimulation2D::Step(float deltaTime)
    {
        if (Particles.empty() || deltaTime <= 0.0f || !std::isfinite(deltaTime)) return;
        deltaTime = std::min(deltaTime, 1.0f / 20.0f);
        const float targetStep = 1.0f / 120.0f;
        const int substeps = std::min(std::max(Settings.MaxSubsteps, 1),
            std::max(1, static_cast<int>(std::ceil(deltaTime / targetStep))));
        const float substep = deltaTime / static_cast<float>(substeps);
        if (Backend != FluidComputeBackend::CPU)
        {
            if (!GpuSolver.Instance) GpuSolver.Instance = Backend == FluidComputeBackend::CUDA ?
                CreateCudaFluidSolver() : Backend == FluidComputeBackend::Metal ? CreateMetalFluidSolver() : CreateDX11FluidSolver();
            if (GpuSolver.Instance && GpuSolver.Instance->Step(Settings, GetKernelRestDensity(),
                substep, substeps, Particles, BackendError))
            {
                Grid.Build(Particles, std::max(Settings.SmoothingRadius, 2.0f * Settings.ParticleRadius));
                return;
            }
            if (BackendError.empty()) BackendError = "GPU backend creation failed";
            Backend = FluidComputeBackend::CPU;
            GpuSolver.Instance.reset();
        }
        for (int index = 0; index < substeps; ++index) Substep(substep);
    }

    bool FluidSimulation2D::SetBackend(FluidComputeBackend backend)
    {
        BackendError.clear();
        const auto& availability = GetOperatorAvailability(GpuOperator::Liquid, backend);
        if (!availability.Available) { BackendError = availability.Message; return false; }
        if (Backend != backend) GpuSolver.Instance.reset();
        Backend = backend;
        return true;
    }

    void FluidSimulation2D::Clear()
    {
        Particles.clear();
        GpuSolver.Instance.reset();
        Pairs.clear();
        Grid.Build(Particles, Settings.SmoothingRadius);
    }

    float FluidSimulation2D::GetKernelRestDensity() const
    {
        if (Settings.RestKernelDensity > 0.0f) return Settings.RestKernelDensity;
        const float radius = std::max(Settings.SmoothingRadius, 2.0f * Settings.ParticleRadius);
        const int extent = static_cast<int>(std::ceil(radius / InitialSpacing));
        float density = 0.0f;
        for (int y = -extent; y <= extent; ++y)
            for (int x = -extent; x <= extent; ++x)
            {
                const float q = std::max(0.0f, 1.0f - InitialSpacing * InitialSpacing *
                    static_cast<float>(x * x + y * y) / (radius * radius));
                density += q * q * q;
            }
        return std::max(density, 1.0f);
    }

    float FluidSimulation2D::SampleOccupancy(const Vector2& position) const
    {
        const float radius = std::max(Settings.SmoothingRadius, 2.0f * Settings.ParticleRadius);
        float weightSum = 0.0f;
        Grid.Query(position, radius, [&](std::size_t index) {
            const float distance = (Particles[index].Position - position).Length();
            if (distance < radius) weightSum += 1.0f - distance / radius;
        });
        return Clamp(weightSum, 0.0f, 1.0f);
    }

    Vector2 FluidSimulation2D::SampleVelocity(const Vector2& position) const
    {
        const float radius = std::max(Settings.SmoothingRadius, 2.0f * Settings.ParticleRadius);
        Vector2 velocity = Vector2::Zero();
        float weightSum = 0.0f;
        Grid.Query(position, radius, [&](std::size_t index) {
            const float distance = (Particles[index].Position - position).Length();
            if (distance >= radius) return;
            const float weight = 1.0f - distance / radius;
            velocity += weight * Particles[index].Velocity;
            weightSum += weight;
        });
        return weightSum > Epsilon ? velocity / weightSum : Vector2::Zero();
    }

    void FluidSimulation2D::BuildPairs(float radius)
    {
        Grid.Build(Particles, radius);
        Pairs.clear();
        const float inverseRadiusSquared = 1.0f / (radius * radius);
        for (std::size_t first = 0; first < Particles.size(); ++first)
        {
            Grid.Query(Particles[first].Position, radius, [&](std::size_t second) {
                if (second <= first) return;
                Vector2 delta = Particles[first].Position - Particles[second].Position;
                const float distanceSquared = delta.SquareLength();
                if (distanceSquared >= radius * radius) return;
                if (distanceSquared < 1.0e-12f)
                    delta = ((first + second) & 1u) ? Vector2{0.001f * radius, 0.0f}
                                                                 : Vector2{0.0f, 0.001f * radius};
                const float q = 1.0f - distanceSquared * inverseRadiusSquared;
                Pairs.push_back({first, second, (-6.0f * inverseRadiusSquared * q * q) * delta, q * q * q});
            });
        }
    }

    void FluidSimulation2D::Substep(float deltaTime)
    {
        const std::size_t count = Particles.size();
        PreviousPositions.resize(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            PreviousPositions[i] = Particles[i].Position;
            Particles[i].Velocity += deltaTime * Settings.Gravity;
            Particles[i].Position += deltaTime * Particles[i].Velocity;
            ConstrainBoundaries(Particles[i]);
        }
        const float radius = std::max(Settings.SmoothingRadius, 2.0f * Settings.ParticleRadius);
        const float inverseRestDensity = 1.0f / GetKernelRestDensity();
        Lambdas.resize(count);
        for (int iteration = 0; iteration < std::max(Settings.PressureIterations, 1); ++iteration)
        {
            // Rebuild after corrections so fast motion cannot leave stale neighbors.
            BuildPairs(radius);
            GradientSums.assign(count, Vector2::Zero());
            GradientSquares.assign(count, 0.0f);
            for (auto& particle : Particles) particle.Density = 1.0f;
            for (const auto& pair : Pairs)
            {
                Particles[pair.First].Density += pair.Weight;
                Particles[pair.Second].Density += pair.Weight;
                const Vector2 gradient = inverseRestDensity * pair.Gradient;
                GradientSums[pair.First] += gradient;
                GradientSums[pair.Second] -= gradient;
                GradientSquares[pair.First] += gradient.SquareLength();
                GradientSquares[pair.Second] += gradient.SquareLength();
            }
            for (std::size_t i = 0; i < count; ++i)
            {
                const float constraint = std::max(Particles[i].Density * inverseRestDensity - 1.0f, 0.0f);
                Lambdas[i] = -constraint / (GradientSquares[i] + GradientSums[i].SquareLength() + 0.01f / (radius * radius));
            }
            VelocityDeltas.assign(count, Vector2::Zero());
            for (const auto& pair : Pairs)
            {
                const Vector2 correction = (Clamp(Settings.PressureStiffness, 0.0f, 1.0f) *
                    (Lambdas[pair.First] + Lambdas[pair.Second]) * inverseRestDensity) * pair.Gradient;
                VelocityDeltas[pair.First] += correction;
                VelocityDeltas[pair.Second] -= correction;
            }
            for (std::size_t i = 0; i < count; ++i)
            {
                const float length = VelocityDeltas[i].Length();
                Particles[i].Position += VelocityDeltas[i] * std::min(1.0f, 0.2f * radius / std::max(length, Epsilon));
                ConstrainBoundaries(Particles[i]);
            }
        }
        for (std::size_t i = 0; i < count; ++i)
            Particles[i].Velocity = (Particles[i].Position - PreviousPositions[i]) / deltaTime;
        UpdateDensities();
        VelocityDeltas.assign(count, Vector2::Zero());
        const float viscosity = 1.0f - std::exp(-std::max(Settings.Viscosity, 0.0f) * deltaTime * 60.0f);
        for (const auto& pair : Pairs)
        {
            const float weight = viscosity * pair.Weight /
                std::max(Particles[pair.First].Density, Particles[pair.Second].Density);
            const Vector2 exchange = weight * (Particles[pair.Second].Velocity - Particles[pair.First].Velocity);
            VelocityDeltas[pair.First] += exchange;
            VelocityDeltas[pair.Second] -= exchange;
        }
        for (std::size_t i = 0; i < count; ++i)
        {
            Particles[i].Velocity += VelocityDeltas[i];
            // Apply wall restitution after reconstructing velocity from corrected positions.
            ConstrainBoundaries(Particles[i]);
        }
    }

    void FluidSimulation2D::ConstrainBoundaries(FluidParticle2D& particle)
    {
        const float radius = Settings.ParticleRadius;
        const Vector2 minimum = Settings.BoundsMinimum + Vector2{radius, radius};
        const Vector2 maximum = Settings.BoundsMaximum - Vector2{radius, radius};
        Vector2 localPosition = Settings.ContainerTransform.InverseTransformPoint(particle.Position);
        Vector2 localVelocity = Settings.ContainerTransform.InverseTransformVector(particle.Velocity);
        if (localPosition.X <= minimum.X)
        {
            localPosition.X = minimum.X;
            if (localVelocity.X < 0.0f) localVelocity.X *= -Settings.BoundaryRestitution;
        }
        else if (localPosition.X >= maximum.X)
        {
            localPosition.X = maximum.X;
            if (localVelocity.X > 0.0f) localVelocity.X *= -Settings.BoundaryRestitution;
        }
        if (localPosition.Y <= minimum.Y)
        {
            localPosition.Y = minimum.Y;
            if (localVelocity.Y < 0.0f) localVelocity.Y *= -Settings.BoundaryRestitution;
        }
        else if (localPosition.Y >= maximum.Y)
        {
            localPosition.Y = maximum.Y;
            if (localVelocity.Y > 0.0f) localVelocity.Y *= -Settings.BoundaryRestitution;
        }
        particle.Position = Settings.ContainerTransform.TransformPoint(localPosition);
        particle.Velocity = Settings.ContainerTransform.TransformVector(localVelocity);
    }

    void FluidSimulation2D::UpdateDensities()
    {
        BuildPairs(std::max(Settings.SmoothingRadius, 2.0f * Settings.ParticleRadius));
        for (auto& particle : Particles) particle.Density = 1.0f;
        for (const auto& pair : Pairs)
        {
            Particles[pair.First].Density += pair.Weight;
            Particles[pair.Second].Density += pair.Weight;
        }
    }
}
