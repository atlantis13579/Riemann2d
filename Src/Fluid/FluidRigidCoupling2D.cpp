#include "FluidRigidCoupling2D.h"

#include "Collision/Box2.h"
#include "Collision/Sphere2.h"
#include "Dynamics/RigidBody2D.h"
#include "Fluid/FluidSimulation2D.h"
#include "Maths/Maths.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace Riemann2D
{
    namespace
    {
        bool IsFinite(const Vector2& value)
        {
            return std::isfinite(value.X) && std::isfinite(value.Y);
        }

        void BuildBoxSamples(const Box2& box, int samplesPerAxis,
                             std::vector<Vector2>& samples)
        {
            samples.clear();
            samples.reserve(static_cast<std::size_t>(samplesPerAxis * samplesPerAxis));
            const Vector2 halfExtent = box.GetHalfExtent();
            for (int y = 0; y < samplesPerAxis; ++y)
            {
                for (int x = 0; x < samplesPerAxis; ++x)
                {
                    const Vector2 fraction{
                        (static_cast<float>(x) + 0.5f) / static_cast<float>(samplesPerAxis),
                        (static_cast<float>(y) + 0.5f) / static_cast<float>(samplesPerAxis)};
                    samples.push_back({(2.0f * fraction.X - 1.0f) * halfExtent.X,
                                       (2.0f * fraction.Y - 1.0f) * halfExtent.Y});
                }
            }
        }

        void BuildSphereSamples(const Sphere2& sphere, int samplesPerAxis,
                                std::vector<Vector2>& samples)
        {
            samples.clear();
            samples.reserve(static_cast<std::size_t>(samplesPerAxis * samplesPerAxis));
            for (int y = 0; y < samplesPerAxis; ++y)
            {
                for (int x = 0; x < samplesPerAxis; ++x)
                {
                    const Vector2 unitPosition{
                        2.0f * (static_cast<float>(x) + 0.5f) /
                            static_cast<float>(samplesPerAxis) - 1.0f,
                        2.0f * (static_cast<float>(y) + 0.5f) /
                            static_cast<float>(samplesPerAxis) - 1.0f};
                    if (unitPosition.SquareLength() <= 1.0f)
                    {
                        samples.push_back(sphere.GetRadius() * unitPosition);
                    }
                }
            }
            if (samples.empty()) samples.push_back(Vector2::Zero());
        }
    }

    FluidRigidCoupling2D::FluidRigidCoupling2D(
        FluidSimulation2D* fluid, const FluidRigidCouplingSettings2D& settings)
        : Fluid(fluid)
        , Settings(settings)
    {
    }

    bool FluidRigidCoupling2D::Evaluate(const RigidBody2D& body,
                                       FluidRigidCouplingResult2D& result) const
    {
        result = {};
        if (Fluid == nullptr || body.GetType() != BodyType2D::Dynamic ||
            Settings.SamplesPerAxis < 1 || Settings.SamplesPerAxis > 16 ||
            Settings.LinearDragCoefficient < 0.0f ||
            Settings.MaximumLinearAcceleration <= 0.0f ||
            Settings.MaximumAngularAcceleration <= 0.0f)
        {
            return false;
        }

        std::vector<Vector2> localSamples;
        float shapeArea = 0.0f;
        if (body.GetShape().GetType() == ShapeType2::Box)
        {
            const auto& box = static_cast<const Box2&>(body.GetShape());
            BuildBoxSamples(box, Settings.SamplesPerAxis, localSamples);
            shapeArea = 4.0f * box.GetHalfExtent().X * box.GetHalfExtent().Y;
        }
        else if (body.GetShape().GetType() == ShapeType2::Sphere2)
        {
            const auto& sphere = static_cast<const Sphere2&>(body.GetShape());
            BuildSphereSamples(sphere, Settings.SamplesPerAxis, localSamples);
            shapeArea = Pi * sphere.GetRadius() * sphere.GetRadius();
        }
        else
        {
            return false;
        }

        const FluidSettings2D& fluidSettings = Fluid->GetSettings();
        if (localSamples.empty() || shapeArea <= 0.0f ||
            fluidSettings.RestDensity <= 0.0f || !IsFinite(fluidSettings.Gravity))
        {
            return false;
        }

        result.SampleCount = static_cast<int>(localSamples.size());
        const float sampleArea = shapeArea / static_cast<float>(localSamples.size());
        for (const Vector2& localPosition : localSamples)
        {
            const Vector2 position = body.GetTransform().TransformPoint(localPosition);
            const float occupancy = Fluid->SampleOccupancy(position);
            if (occupancy <= Epsilon) continue;

            const float displacedArea = sampleArea * occupancy;
            const float displacedMass = fluidSettings.RestDensity * displacedArea;
            const Vector2 radius = position - body.GetTransform().Position;
            const Vector2 bodyVelocity = body.GetLinearVelocity() +
                                         Cross(body.GetAngularVelocity(), radius);
            const Vector2 relativeVelocity = Fluid->SampleVelocity(position) - bodyVelocity;
            const Vector2 buoyancyForce = -displacedMass * fluidSettings.Gravity;
            const Vector2 dragForce = Settings.LinearDragCoefficient * displacedMass *
                                      relativeVelocity;
            const Vector2 sampleForce = buoyancyForce + dragForce;
            if (!IsFinite(sampleForce)) return false;

            result.Force += sampleForce;
            result.Torque += radius.Cross(sampleForce);
            result.BuoyancyForce += buoyancyForce;
            result.DragForce += dragForce;
            result.DisplacedMass += displacedMass;
            result.SubmergedArea += displacedArea;
            ++result.ActiveSampleCount;
        }

        float clampScale = 1.0f;
        const float linearAcceleration = result.Force.Length() * body.GetInverseMass();
        if (linearAcceleration > Settings.MaximumLinearAcceleration)
        {
            clampScale = std::min(clampScale,
                Settings.MaximumLinearAcceleration / linearAcceleration);
        }
        const float angularAcceleration =
            std::abs(result.Torque) * body.GetInverseInertia();
        if (angularAcceleration > Settings.MaximumAngularAcceleration)
        {
            clampScale = std::min(clampScale,
                Settings.MaximumAngularAcceleration / angularAcceleration);
        }
        if (!std::isfinite(clampScale)) return false;

        result.ClampScale = clampScale;
        result.Force *= clampScale;
        result.Torque *= clampScale;
        result.BuoyancyForce *= clampScale;
        result.DragForce *= clampScale;
        return true;
    }

    bool FluidRigidCoupling2D::ApplyForce(RigidBody2D& body)
    {
        FluidRigidCouplingResult2D result;
        if (!Evaluate(body, result)) return false;
        body.ApplyForce(result.Force);
        body.ApplyTorque(result.Torque);
        return true;
    }
}
