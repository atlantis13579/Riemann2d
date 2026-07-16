#pragma once

#include "Dynamics/ForceField2D.h"
#include "Maths/Vector2.h"

namespace Riemann2D
{
    class FluidSimulation2D;
    class RigidBody2D;

    struct FluidRigidCouplingSettings2D
    {
        int SamplesPerAxis = 4;
        float LinearDragCoefficient = 2.0f;
        float MaximumLinearAcceleration = 80.0f;
        float MaximumAngularAcceleration = 120.0f;
    };

    struct FluidRigidCouplingResult2D
    {
        Vector2 Force = Vector2::Zero();
        float Torque = 0.0f;
        Vector2 BuoyancyForce = Vector2::Zero();
        Vector2 DragForce = Vector2::Zero();
        float DisplacedMass = 0.0f;
        float SubmergedArea = 0.0f;
        float ClampScale = 1.0f;
        int SampleCount = 0;
        int ActiveSampleCount = 0;
    };

    // The fluid is externally owned and must outlive this force field.
    class FluidRigidCoupling2D final : public ForceField2D
    {
    public:
        explicit FluidRigidCoupling2D(
            FluidSimulation2D* fluid,
            const FluidRigidCouplingSettings2D& settings = {});

        bool ApplyForce(RigidBody2D& body) override;
        bool Evaluate(const RigidBody2D& body, FluidRigidCouplingResult2D& result) const;

        FluidSimulation2D* GetFluid() const { return Fluid; }
        const FluidRigidCouplingSettings2D& GetSettings() const { return Settings; }

    private:
        FluidSimulation2D* Fluid = nullptr;
        FluidRigidCouplingSettings2D Settings;
    };
}
