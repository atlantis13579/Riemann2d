#include "RopeSimulation2D.h"

#include "Maths/Maths.h"

#include <algorithm>
#include <cmath>

namespace Riemann2D
{
    void RopeSimulation2D::CreateLine(const Vector2& start, const Vector2& end,
                                          int particleCount, float particleMass)
    {
        Clear();
        particleCount = std::max(particleCount, 2);
        particleMass = std::max(particleMass, Epsilon);
        Particles.resize(static_cast<std::size_t>(particleCount));
        for (int index = 0; index < particleCount; ++index)
        {
            const float fraction = static_cast<float>(index) / static_cast<float>(particleCount - 1);
            RopeParticle2D& particle = Particles[static_cast<std::size_t>(index)];
            particle.Position = start + fraction * (end - start);
            particle.PreviousPosition = particle.Position;
            particle.InverseMass = 1.0f / particleMass;
        }
        RestLengths.resize(static_cast<std::size_t>(particleCount - 1));
        StretchLambdas.resize(RestLengths.size());
        for (int index = 0; index + 1 < particleCount; ++index)
        {
            RestLengths[static_cast<std::size_t>(index)] =
                (Particles[static_cast<std::size_t>(index + 1)].Position -
                 Particles[static_cast<std::size_t>(index)].Position).Length();
        }
        BendRestLengths.resize(static_cast<std::size_t>(particleCount - 2));
        BendLambdas.resize(BendRestLengths.size());
        for (int index = 0; index + 2 < particleCount; ++index)
        {
            BendRestLengths[static_cast<std::size_t>(index)] =
                (Particles[static_cast<std::size_t>(index + 2)].Position -
                 Particles[static_cast<std::size_t>(index)].Position).Length();
        }
    }

    void RopeSimulation2D::Clear()
    {
        Particles.clear();
        RestLengths.clear();
        BendRestLengths.clear();
        StretchLambdas.clear();
        BendLambdas.clear();
    }

    void RopeSimulation2D::PinParticle(int index, const Vector2& position)
    {
        if (index < 0 || index >= static_cast<int>(Particles.size())) return;
        RopeParticle2D& particle = Particles[static_cast<std::size_t>(index)];
        particle.Position = position;
        particle.PreviousPosition = position;
        particle.Velocity = Vector2::Zero();
        particle.InverseMass = 0.0f;
    }

    void RopeSimulation2D::SetParticlePosition(int index, const Vector2& position)
    {
        if (index < 0 || index >= static_cast<int>(Particles.size())) return;
        RopeParticle2D& particle = Particles[static_cast<std::size_t>(index)];
        particle.Position = position;
        if (particle.InverseMass <= 0.0f)
        {
            particle.PreviousPosition = position;
            particle.Velocity = Vector2::Zero();
        }
    }

    void RopeSimulation2D::SolveDistance(int first, int second, float restLength,
                                             float compliance, float& lambda,
                                             float inverseDeltaTimeSquared)
    {
        RopeParticle2D& a = Particles[static_cast<std::size_t>(first)];
        RopeParticle2D& b = Particles[static_cast<std::size_t>(second)];
        const Vector2 delta = b.Position - a.Position;
        const float length = delta.Length();
        const float inverseMass = a.InverseMass + b.InverseMass;
        if (length <= Epsilon || inverseMass <= Epsilon) return;
        const float alpha = std::max(compliance, 0.0f) * inverseDeltaTimeSquared;
        const float deltaLambda = -(length - restLength + alpha * lambda) / (inverseMass + alpha);
        lambda += deltaLambda;
        const Vector2 correction = deltaLambda * (delta / length);
        a.Position -= a.InverseMass * correction;
        b.Position += b.InverseMass * correction;
    }

    void RopeSimulation2D::Step(float deltaTime)
    {
        if (Particles.size() < 2 || deltaTime <= 0.0f) return;
        deltaTime = std::min(deltaTime, 1.0f / 30.0f);
        const float damping = std::exp(-std::max(Settings.Damping, 0.0f) * deltaTime);
        for (RopeParticle2D& particle : Particles)
        {
            particle.PreviousPosition = particle.Position;
            if (particle.InverseMass > 0.0f)
            {
                particle.Velocity = damping * (particle.Velocity + deltaTime * Settings.Gravity);
                particle.Position += deltaTime * particle.Velocity;
            }
        }
        std::fill(StretchLambdas.begin(), StretchLambdas.end(), 0.0f);
        std::fill(BendLambdas.begin(), BendLambdas.end(), 0.0f);
        const float inverseDeltaTimeSquared = 1.0f / (deltaTime * deltaTime);
        for (int iteration = 0; iteration < std::max(Settings.Iterations, 1); ++iteration)
        {
            for (std::size_t index = 0; index < BendRestLengths.size(); ++index)
            {
                SolveDistance(static_cast<int>(index), static_cast<int>(index + 2),
                    BendRestLengths[index], Settings.BendCompliance, BendLambdas[index],
                    inverseDeltaTimeSquared);
            }
            for (std::size_t index = 0; index < RestLengths.size(); ++index)
            {
                SolveDistance(static_cast<int>(index), static_cast<int>(index + 1),
                    RestLengths[index], Settings.StretchCompliance, StretchLambdas[index],
                    inverseDeltaTimeSquared);
            }
        }
        for (RopeParticle2D& particle : Particles)
        {
            if (particle.InverseMass > 0.0f)
            {
                particle.Velocity = (particle.Position - particle.PreviousPosition) / deltaTime;
            }
        }
    }
}
