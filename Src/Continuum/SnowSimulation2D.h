#pragma once

#include "Continuum/MpmMaterial2D.h"
#include "Maths/Vector2.h"

#include <memory>
#include <string>
#include <vector>

namespace Riemann2D
{
    enum class ContinuumBackend { CPU, SIMD, CUDA, ComputeShader, Metal };
    enum class MpmTransfer2D { MPM, APIC, MLS };
    enum class ContinuumQuality { Realtime, High };

    struct SnowParticle2D
    {
        Vector2 Position;
        Vector2 Velocity;
        Continuum::Matrix2 Elastic = Continuum::Matrix2::Identity();
        Continuum::Matrix2 Affine;
        float PlasticVolume = 1;
        float Volume = 0;
        float Mass = 0;
        float Fixed = 0;
    };

    struct SnowCapsule2D
    {
        Vector2 Center{0, 0.8f};
        Vector2 Velocity;
        float HalfLength = 0.6f;
        float Radius = 0.45f;
        float Friction = 0.55f;
        bool Enabled = true;
    };

    struct SnowSettings2D
    {
        Vector2 BoundsMinimum{-8, -0.5f};
        Vector2 BoundsMaximum{8, 9.5f};
        Vector2 Gravity{0, -9.81f};
        float CellSize = 0.2f;
        float Density = 200;
        float FlipRatio = 0.95f;
        // A compacted, stationary lower bed supports the deformable surface layer.
        float FixedBedHeight = -1000000.0f;
        float FrameBudgetMilliseconds = 0;
        int MaximumSubsteps = 24;
        MpmTransfer2D Transfer = MpmTransfer2D::MLS;
        Continuum::Material2D Material;
    };

    struct SnowStatistics2D
    {
        int Substeps = 0;
        double TotalMass = 0;
        Vector2 Momentum;
        float MaximumSpeed = 0;
        float SimulatedTime = 0;
        float PendingTime = 0;
        bool TimeLimited = false;
    };

    class SnowGpuSolver2D;

    class SnowSimulation2D
    {
    public:
        explicit SnowSimulation2D(const SnowSettings2D& settings = {});
        ~SnowSimulation2D();
        SnowSimulation2D(const SnowSimulation2D&) = delete;
        SnowSimulation2D& operator=(const SnowSimulation2D&) = delete;

        void Reset(ContinuumQuality quality = ContinuumQuality::Realtime);
        void Clear();
        void AddRectangle(Vector2 minimum, Vector2 maximum, float spacing, Vector2 velocity = {});
        void Step(float deltaTime);
        bool SetBackend(ContinuumBackend backend);
        ContinuumBackend GetBackend() const { return Backend; }
        const std::string& GetBackendError() const { return BackendError; }
        SnowSettings2D& GetSettings() { return Settings; }
        const SnowSettings2D& GetSettings() const { return Settings; }
        const std::vector<SnowParticle2D>& GetParticles() const { return Particles; }
        const SnowStatistics2D& GetStatistics() const { return Statistics; }
        const SnowCapsule2D& GetCapsule() const { return Capsule; }
        SnowCapsule2D& GetCapsule() { return Capsule; }
        void SetCapsuleTarget(float x) { CapsuleTarget = x; }
        float GetSurfaceHeight() const { return SurfaceHeight; }
        float GetScrapeDepth() const;
        static bool IsSimdAvailable();

    private:
        struct GridNode
        {
            Vector2 Momentum;
            Vector2 Force;
            Vector2 OldVelocity;
            float Mass = 0;
        };
        SnowSettings2D Settings;
        SnowCapsule2D Capsule;
        SnowStatistics2D Statistics;
        std::vector<SnowParticle2D> Particles;
        std::vector<GridNode> Grid;
        ContinuumBackend Backend = ContinuumBackend::CPU;
        std::string BackendError;
        std::unique_ptr<SnowGpuSolver2D> GpuSolver;
        int Width = 0, Height = 0;
        float PendingTime = 0;
        float CapsuleTarget = 0;
        bool FollowSurface = false;
        float SurfaceHeight = 1.8f, SurfaceSampleX = -1000;
        int SurfaceDirection = 0;
        void UpdateSurfaceTarget();

        void ValidateSettings() const;
        void PrepareGrid();
        void StepCpu(float deltaTime, bool simd);
        void GatherStatistics();
        Vector2 Collide(Vector2 position, Vector2 velocity, float padding) const;
    };
}
