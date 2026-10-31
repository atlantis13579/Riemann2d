#pragma once

#include "Maths/Vector2.h"
#include "GpuFluidSolvers.h"

#include <vector>

namespace Riemann2D
{
    struct AerodynamicsSettings2D
    {
        float VelocityDiffusion = 0.001f;
        float DensityDissipation = 0.08f;
        // Eight fine-grid smoothing passes per multigrid cycle.
        int PressureIterations = 24;
        float MaximumTimeStep = 1.0f / 30.0f;
        float VorticityStrength = 1.5f;
        bool UseMacCormack = true;
    };

    class AerodynamicsSimulation2D
    {
    public:
        bool Initialize(const Vector2& minimum, const Vector2& maximum, int width, int height);
        void Clear();
        void AddDensity(const Vector2& position, float amount);
        void AddVelocity(const Vector2& position, const Vector2& velocity);
        void AddEmitter(const Vector2& position, float radius, float densityRate,
                        const Vector2& velocity, float deltaTime);
        void Step(float deltaTime);
        void ProjectVelocity();
        bool SetBackend(FluidComputeBackend backend);
        FluidComputeBackend GetBackend() const { return Backend; }
        const std::string& GetBackendError() const { return BackendError; }

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
        const std::vector<float>& GetDensityField() const { return Density; }
        AerodynamicsSettings2D& GetSettings() { return Settings; }

    private:
        struct PressureLevel
        {
            int Width = 0;
            int Height = 0;
            float WeightX = 1.0f;
            float WeightY = 1.0f;
            std::vector<float> Pressure;
            std::vector<float> RightHandSide;
            std::vector<float> Residual;
            std::vector<float> InverseDiagonal;
        };

        Vector2 SampleGridVelocity(float x, float y) const;
        void AdvectVelocity(float deltaTime);
        bool StepGpu(float deltaTime, bool projectOnly);
        void ProjectVelocityCPU();
        void AdvectDensity(float deltaTime);
        void ApplyVorticity(float deltaTime);
        void EnforceBoundary();
        void MultigridCycle(std::size_t level);
        static void SmoothPressure(PressureLevel& level, int iterations);

        AerodynamicsSettings2D Settings;
        Vector2 BoundsMinimum = Vector2::Zero();
        Vector2 BoundsMaximum = Vector2::Zero();
        Vector2 CellSize{1.0f, 1.0f};
        Vector2 InverseCellSize{1.0f, 1.0f};
        int Width = 0;
        int Height = 0;
        std::vector<float> Density;
        std::vector<float> DensityScratch;
        std::vector<float> DensityForward;
        // MAC layout: U on vertical faces, V on horizontal faces.
        std::vector<float> U;
        std::vector<float> V;
        std::vector<float> UScratch;
        std::vector<float> VScratch;
        std::vector<Vector2> CenterVelocity;
        std::vector<float> Curl;
        std::vector<Vector2> VorticityForce;
        std::vector<PressureLevel> PressureLevels;
        FluidComputeBackend Backend = FluidComputeBackend::CPU;
        std::string BackendError;
        ComputeResource<AirGpuSolver> GpuSolver;
    };
}
