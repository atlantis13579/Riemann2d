#pragma once

#include "Maths/Vector2.h"
#include "GpuFluidSolvers.h"

#include <functional>
#include <vector>

namespace Riemann2D
{
    class PhysicsWorld2D;
    class RigidBody2D;

    struct AerodynamicSurfaceSample2D
    {
        float Pressure = 0.0f;
        float Density = 0.0f;
        Vector2 Velocity = Vector2::Zero();
    };

    using AerodynamicSurfaceSampler2D = std::function<AerodynamicSurfaceSample2D(const Vector2&)>;

    struct AerodynamicForceSettings2D
    {
        int SurfaceSamples = 48;
        // Resolved obstacle pressure already includes incident-flow loading.
        float DragCoefficient = 0.0f;
        float MaximumLinearAcceleration = 80.0f;
        float MaximumAngularAcceleration = 120.0f;
    };

    struct AerodynamicForceResult2D
    {
        Vector2 Force = Vector2::Zero();
        Vector2 PressureForce = Vector2::Zero();
        Vector2 DragForce = Vector2::Zero();
        float Torque = 0.0f;
        float ClampScale = 1.0f;
        int SampleCount = 0;
    };

    struct AerodynamicsSettings2D
    {
        float AirDensity = 1.225f;
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
        // The externally owned world must outlive boundary and force coupling.
        void SetRigidBodies(PhysicsWorld2D* world);
        void SynchronizeRigidBodies();
        // Surface traction is integrated per unit out-of-plane depth.
        bool EvaluateBodyForce(const RigidBody2D& body, AerodynamicForceResult2D& result) const;
        static bool EvaluateSurfaceForce(const RigidBody2D& body,
                                         const AerodynamicSurfaceSampler2D& sampler,
                                         const AerodynamicForceSettings2D& settings,
                                         AerodynamicForceResult2D& result);
        AerodynamicForceSettings2D& GetForceSettings() { return ForceSettings; }
        const AerodynamicForceSettings2D& GetForceSettings() const { return ForceSettings; }
        float GetMaximumBoundaryVelocityError() const { return RigidBoundary.MaximumBoundaryError(U, V); }
        void AddDensity(const Vector2& position, float amount);
        void AddVelocity(const Vector2& position, const Vector2& velocity);
        void AddEmitter(const Vector2& position, float radius, float densityRate,
                        const Vector2& velocity, float deltaTime);
        // Advances air and applies surface forces before the physics world steps.
        void Step(float deltaTime);
        void ProjectVelocity(float deltaTime = 1.0f / 60.0f);
        bool SetBackend(FluidComputeBackend backend);
        FluidComputeBackend GetBackend() const { return Backend; }
        const std::string& GetBackendError() const { return BackendError; }

        float SampleDensity(const Vector2& position) const;
        Vector2 SampleVelocity(const Vector2& position) const;
        // Gauge pressure in force per unit area; zero outside the simulation domain.
        float SamplePressure(const Vector2& position) const;
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
        const AerodynamicsSettings2D& GetSettings() const { return Settings; }

    private:
        void ApplyRigidBodyForces();
        // Geometry input for all backends and the CPU boundary solver.
        class RigidBoundaryState
        {
        public:
            void Update(const PhysicsWorld2D* world, Vector2 minimum, Vector2 cellSize,
                        int width, int height);
            void DisplaceDensity(std::vector<float>& density) const;
            bool Active() const;
            AirBoundaryData GetGpuData() const { return {Solid, DisplacementHead, DisplacementNext, BlockU, BlockV, VelocityU, VelocityV}; }
            bool IsSolid(int x, int y) const;
            Vector2 ClipTrace(Vector2 start, Vector2 end) const;
            float ExteriorValue(const std::vector<float>& values, int index) const;
            void EnforceVelocity(std::vector<float>& u, std::vector<float>& v) const;
            void Project(std::vector<float>& u, std::vector<float>& v, std::vector<float>& pressure);
            void RemoveInteriorDensity(std::vector<float>& density) const;
            float MaximumBoundaryError(const std::vector<float>& u, const std::vector<float>& v) const;

        private:
            int Width = 0, Height = 0, SolidCount = 0;
            Vector2 CellSize{1,1};
            std::vector<int> Solid, NearestFluid, DisplacementHead, DisplacementNext, BlockU, BlockV;
            std::vector<float> VelocityU, VelocityV;
            std::vector<double> Diagonal, Correction, Residual, Direction, Product, Preconditioned;
            std::vector<double> RightWeight, UpWeight, Factor, Forward;
        };

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
        bool StepGpu(float deltaTime, bool projectOnly, bool synchronizeOnly = false);
        void ProjectVelocityCPU();
        void AdvectDensity(float deltaTime);
        void ApplyVorticity(float deltaTime);
        void EnforceBoundary();
        void MultigridCycle(std::size_t level);
        static void SmoothPressure(PressureLevel& level, int iterations);

        AerodynamicsSettings2D Settings;
        AerodynamicForceSettings2D ForceSettings;
        Vector2 BoundsMinimum = Vector2::Zero();
        Vector2 BoundsMaximum = Vector2::Zero();
        Vector2 CellSize{1.0f, 1.0f};
        Vector2 InverseCellSize{1.0f, 1.0f};
        int Width = 0;
        int Height = 0;
        float ProjectionTimeStep = 0.0f;
        PhysicsWorld2D* RigidWorld = nullptr;
        RigidBoundaryState RigidBoundary;
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
