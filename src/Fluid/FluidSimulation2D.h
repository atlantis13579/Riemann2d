#pragma once

#include "Maths/Transform2.h"
#include "ParticleGrid2D.h"
#include "GpuFluidSolvers.h"

#include <vector>

namespace Riemann2D
{
    struct FluidParticle2D
    {
        Vector2 Position = Vector2::Zero();
        Vector2 Velocity = Vector2::Zero();
        float Density = 0.0f;
    };

    struct FluidSettings2D
    {
        Vector2 BoundsMinimum{-7.0f, -2.0f};
        Vector2 BoundsMaximum{7.0f, 9.0f};
        Transform2 ContainerTransform;
        Vector2 Gravity{0.0f, -9.81f};
        float ParticleRadius = 0.05f;
        float SmoothingRadius = 0.34f;
        float PressureStiffness = 0.9f;
        float Viscosity = 0.08f;
        float RestDensity = 1.25f;
        // Zero calibrates the dimensionless kernel sum from the initial spacing.
        float RestKernelDensity = 0.0f;
        float BoundaryRestitution = 0.15f;
        int PressureIterations = 4;
        int MaxSubsteps = 4;
    };

    class FluidSimulation2D
    {
    public:
        void CreateBlock(const Vector2& minimum, const Vector2& maximum, float spacing);
        void Clear();
        void Step(float deltaTime);
        bool SetBackend(FluidComputeBackend backend);
        FluidComputeBackend GetBackend() const { return Backend; }
        const std::string& GetBackendError() const { return BackendError; }

        FluidSettings2D& GetSettings() { return Settings; }
        const FluidSettings2D& GetSettings() const { return Settings; }
        const std::vector<FluidParticle2D>& GetParticles() const { return Particles; }
        float SampleOccupancy(const Vector2& position) const;
        Vector2 SampleVelocity(const Vector2& position) const;
        float GetKernelRestDensity() const;
        float GetParticleSpacing() const { return InitialSpacing; }

    private:
        void Substep(float deltaTime);
        void ConstrainBoundaries(FluidParticle2D& particle);
        void UpdateDensities();
        void BuildPairs(float radius);

        struct NeighborPair
        {
            std::size_t First;
            std::size_t Second;
            Vector2 Gradient;
            float Weight;
        };

        FluidSettings2D Settings;
        std::vector<FluidParticle2D> Particles;
        std::vector<Vector2> PreviousPositions;
        std::vector<Vector2> VelocityDeltas;
        std::vector<Vector2> GradientSums;
        std::vector<float> GradientSquares;
        std::vector<float> Lambdas;
        std::vector<NeighborPair> Pairs;
        ParticleGrid2D Grid;
        float InitialSpacing = 0.3f;
        FluidComputeBackend Backend = FluidComputeBackend::CPU;
        std::string BackendError;
        ComputeResource<FluidGpuSolver> GpuSolver;
    };
}
