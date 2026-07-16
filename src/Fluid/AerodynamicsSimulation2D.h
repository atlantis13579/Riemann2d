#pragma once

#include "Maths/Vector2.h"

#include <vector>

namespace Riemann2D
{
    struct AerodynamicsSettings2D
    {
        float VelocityDiffusion = 0.001f;
        float DensityDissipation = 0.08f;
        int PressureIterations = 32;
        float MaximumTimeStep = 1.0f / 30.0f;
    };

    class AerodynamicsSimulation2D
    {
    public:
        bool Initialize(const Vector2& minimum, const Vector2& maximum, int width, int height);
        void Clear();
        void AddDensity(const Vector2& position, float amount);
        void AddVelocity(const Vector2& position, const Vector2& velocity);
        void Step(float deltaTime);

        float SampleDensity(const Vector2& position) const;
        Vector2 SampleVelocity(const Vector2& position) const;
        float GetDensity(int x, int y) const;
        Vector2 GetVelocity(int x, int y) const;
        Vector2 GetCellCenter(int x, int y) const;
        float ComputeMeanAbsoluteDivergence() const;

        int GetWidth() const { return Width; }
        int GetHeight() const { return Height; }
        const Vector2& GetBoundsMinimum() const { return BoundsMinimum; }
        const Vector2& GetBoundsMaximum() const { return BoundsMaximum; }
        AerodynamicsSettings2D& GetSettings() { return Settings; }

    private:
        int Index(int x, int y) const;
        float SampleScalar(const std::vector<float>& field, const Vector2& position) const;
        Vector2 SampleVector(const std::vector<Vector2>& field, const Vector2& position) const;
        void DiffuseVelocity(float deltaTime);
        void ProjectVelocity();
        void AdvectVelocity(float deltaTime);
        void AdvectDensity(float deltaTime);
        void EnforceBoundary();

        AerodynamicsSettings2D Settings;
        Vector2 BoundsMinimum = Vector2::Zero();
        Vector2 BoundsMaximum = Vector2::Zero();
        Vector2 CellSize{1.0f, 1.0f};
        int Width = 0;
        int Height = 0;
        std::vector<float> Density;
        std::vector<float> DensityScratch;
        std::vector<float> Pressure;
        std::vector<float> PressureScratch;
        std::vector<float> Divergence;
        std::vector<Vector2> Velocity;
        std::vector<Vector2> VelocityScratch;
    };
}
