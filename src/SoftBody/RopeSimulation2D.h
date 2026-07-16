#pragma once

#include "Maths/Vector2.h"

#include <vector>

namespace Riemann2D
{
    struct RopeParticle2D
    {
        Vector2 Position = Vector2::Zero();
        Vector2 PreviousPosition = Vector2::Zero();
        Vector2 Velocity = Vector2::Zero();
        float InverseMass = 1.0f;
    };

    struct RopeSettings2D
    {
        Vector2 Gravity{0.0f, -9.81f};
        float Damping = 0.02f;
        float StretchCompliance = 1.0e-7f;
        float BendCompliance = 2.0e-4f;
        int Iterations = 12;
    };

    class RopeSimulation2D
    {
    public:
        void CreateLine(const Vector2& start, const Vector2& end, int particleCount,
                        float particleMass = 1.0f);
        void Clear();
        void PinParticle(int index, const Vector2& position);
        void SetParticlePosition(int index, const Vector2& position);
        void Step(float deltaTime);

        const std::vector<RopeParticle2D>& GetParticles() const { return Particles; }
        RopeSettings2D& GetSettings() { return Settings; }
        const RopeSettings2D& GetSettings() const { return Settings; }

    private:
        void SolveDistance(int first, int second, float restLength, float compliance,
                           float& lambda, float inverseDeltaTimeSquared);

        RopeSettings2D Settings;
        std::vector<RopeParticle2D> Particles;
        std::vector<float> RestLengths;
        std::vector<float> BendRestLengths;
        std::vector<float> StretchLambdas;
        std::vector<float> BendLambdas;
    };
}
