#include "FluidSimulation2D.h"

#include "Maths/Maths.h"

#include <algorithm>
#include <cmath>

namespace Riemann2D
{
    void FluidSimulation2D::CreateBlock(const Vector2& minimum, const Vector2& maximum, float spacing)
    {
        Clear();
        spacing = std::max(spacing, 2.0f * Settings.ParticleRadius);
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
        if (Particles.empty() || deltaTime <= 0.0f) return;
        deltaTime = std::min(deltaTime, 1.0f / 20.0f);
        const float targetStep = 1.0f / 120.0f;
        const int substeps = std::min(Settings.MaxSubsteps,
            std::max(1, static_cast<int>(std::ceil(deltaTime / targetStep))));
        const float substep = deltaTime / static_cast<float>(substeps);
        for (int index = 0; index < substeps; ++index) Substep(substep);
    }

    float FluidSimulation2D::SampleOccupancy(const Vector2& position) const
    {
        const float radius = std::max(Settings.SmoothingRadius,
                                      2.0f * Settings.ParticleRadius);
        float weightSum = 0.0f;
        for (const FluidParticle2D& particle : Particles)
        {
            const float distance = (particle.Position - position).Length();
            if (distance < radius) weightSum += 1.0f - distance / radius;
        }
        return Clamp(weightSum, 0.0f, 1.0f);
    }

    Vector2 FluidSimulation2D::SampleVelocity(const Vector2& position) const
    {
        const float radius = std::max(Settings.SmoothingRadius,
                                      2.0f * Settings.ParticleRadius);
        Vector2 weightedVelocity = Vector2::Zero();
        float weightSum = 0.0f;
        for (const FluidParticle2D& particle : Particles)
        {
            const float distance = (particle.Position - position).Length();
            if (distance >= radius) continue;
            const float weight = 1.0f - distance / radius;
            weightedVelocity += weight * particle.Velocity;
            weightSum += weight;
        }
        return weightSum > Epsilon ? weightedVelocity / weightSum : Vector2::Zero();
    }

    void FluidSimulation2D::Substep(float deltaTime)
    {
        PreviousPositions.resize(Particles.size());
        for (std::size_t index = 0; index < Particles.size(); ++index)
        {
            FluidParticle2D& particle = Particles[index];
            PreviousPositions[index] = particle.Position;
            particle.Velocity += deltaTime * Settings.Gravity;
            particle.Position += deltaTime * particle.Velocity;
            ConstrainBoundaries(particle);
        }

        const float radius = std::max(Settings.SmoothingRadius, 2.0f * Settings.ParticleRadius);
        for (int iteration = 0; iteration < std::max(Settings.PressureIterations, 1); ++iteration)
        {
            for (std::size_t first = 0; first < Particles.size(); ++first)
            {
                for (std::size_t second = first + 1; second < Particles.size(); ++second)
                {
                    Vector2 delta = Particles[second].Position - Particles[first].Position;
                    const float distanceSquared = delta.SquareLength();
                    if (distanceSquared >= radius * radius) continue;
                    float distance = std::sqrt(std::max(distanceSquared, Epsilon * Epsilon));
                    if (distanceSquared <= Epsilon * Epsilon)
                    {
                        delta = ((first + second) & 1u) != 0u ? Vector2::UnitX() : Vector2::UnitY();
                        distance = Epsilon;
                    }
                    const float overlap = radius - distance;
                    const Vector2 correction = 0.5f * Settings.PressureStiffness *
                                               overlap * (delta / distance);
                    Particles[first].Position -= correction;
                    Particles[second].Position += correction;
                }
            }
            for (FluidParticle2D& particle : Particles) ConstrainBoundaries(particle);
        }

        for (std::size_t index = 0; index < Particles.size(); ++index)
        {
            Particles[index].Velocity = (Particles[index].Position - PreviousPositions[index]) / deltaTime;
        }
        VelocityDeltas.assign(Particles.size(), Vector2::Zero());
        for (std::size_t first = 0; first < Particles.size(); ++first)
        {
            for (std::size_t second = first + 1; second < Particles.size(); ++second)
            {
                const float distance = (Particles[second].Position - Particles[first].Position).Length();
                if (distance >= radius) continue;
                const float weight = Settings.Viscosity * (1.0f - distance / radius);
                const Vector2 exchange = weight * (Particles[second].Velocity - Particles[first].Velocity);
                VelocityDeltas[first] += exchange;
                VelocityDeltas[second] -= exchange;
            }
        }
        for (std::size_t index = 0; index < Particles.size(); ++index)
        {
            Particles[index].Velocity += VelocityDeltas[index];
        }
        UpdateDensities();
    }

    void FluidSimulation2D::ConstrainBoundaries(FluidParticle2D& particle)
    {
        const float radius = Settings.ParticleRadius;
        const Vector2 minimum = Settings.BoundsMinimum + Vector2{radius, radius};
        const Vector2 maximum = Settings.BoundsMaximum - Vector2{radius, radius};
        Vector2 localPosition = Settings.ContainerTransform.InverseTransformPoint(particle.Position);
        Vector2 localVelocity = Settings.ContainerTransform.InverseTransformVector(particle.Velocity);
        if (localPosition.X < minimum.X)
        {
            localPosition.X = minimum.X;
            localVelocity.X = std::abs(localVelocity.X) * Settings.BoundaryRestitution;
        }
        else if (localPosition.X > maximum.X)
        {
            localPosition.X = maximum.X;
            localVelocity.X = -std::abs(localVelocity.X) * Settings.BoundaryRestitution;
        }
        if (localPosition.Y < minimum.Y)
        {
            localPosition.Y = minimum.Y;
            localVelocity.Y = std::abs(localVelocity.Y) * Settings.BoundaryRestitution;
        }
        else if (localPosition.Y > maximum.Y)
        {
            localPosition.Y = maximum.Y;
            localVelocity.Y = -std::abs(localVelocity.Y) * Settings.BoundaryRestitution;
        }
        particle.Position = Settings.ContainerTransform.TransformPoint(localPosition);
        particle.Velocity = Settings.ContainerTransform.TransformVector(localVelocity);
    }

    void FluidSimulation2D::UpdateDensities()
    {
        const float radius = std::max(Settings.SmoothingRadius, Epsilon);
        for (std::size_t first = 0; first < Particles.size(); ++first)
        {
            float density = 1.0f;
            for (std::size_t second = 0; second < Particles.size(); ++second)
            {
                if (first == second) continue;
                const float distance = (Particles[second].Position - Particles[first].Position).Length();
                if (distance < radius)
                {
                    const float weight = 1.0f - distance / radius;
                    density += weight * weight * weight;
                }
            }
            Particles[first].Density = density;
        }
    }
}
