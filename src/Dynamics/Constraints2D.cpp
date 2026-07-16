#include "Constraints2D.h"

#include <algorithm>
#include <cmath>

namespace Riemann2D
{
    namespace
    {
        constexpr float DefaultStep = 1.0f / 60.0f;

        float SafeStep(float dt)
        {
            return dt > Epsilon ? dt : DefaultStep;
        }

        float WrapAngle(float angle)
        {
            if (!std::isfinite(angle)) return 0.0f;
            return std::remainder(angle, 2.0f * Pi);
        }

        float SoftError(float error, float slop)
        {
            if (std::abs(error) <= slop) return 0.0f;
            return error - std::copysign(slop, error);
        }

        Vector2 WorldAnchor(const RigidBody2D* body, const Vector2& localAnchor)
        {
            return body ? body->GetTransform().TransformPoint(localAnchor) : localAnchor;
        }

        Vector2 PointVelocity(const RigidBody2D* body, const Vector2& worldPoint)
        {
            if (!body) return Vector2::Zero();
            return body->GetLinearVelocity() +
                Cross(body->GetAngularVelocity(), worldPoint - body->GetTransform().Position);
        }

        bool IsDynamic(const RigidBody2D* body)
        {
            return body && body->GetType() == BodyType2D::Dynamic;
        }

        void ApplyPairImpulse(RigidBody2D* bodyA, RigidBody2D* bodyB,
            const Vector2& impulse, const Vector2& pointA, const Vector2& pointB)
        {
            if (bodyA) bodyA->ApplyImpulse(-impulse, pointA);
            if (bodyB) bodyB->ApplyImpulse(impulse, pointB);
        }

        void ApplyPairAngularImpulse(RigidBody2D* bodyA, RigidBody2D* bodyB, float impulse)
        {
            if (bodyA) bodyA->ApplyAngularImpulse(-impulse);
            if (bodyB) bodyB->ApplyAngularImpulse(impulse);
        }

        float AxisMass(const RigidBody2D* bodyA, const RigidBody2D* bodyB,
            const Vector2& pointA, const Vector2& pointB, const Vector2& axis)
        {
            const Vector2 ra = bodyA ? pointA - bodyA->GetTransform().Position : Vector2::Zero();
            const Vector2 rb = bodyB ? pointB - bodyB->GetTransform().Position : Vector2::Zero();
            const float ja = ra.Cross(axis);
            const float jb = rb.Cross(axis);
            return (bodyA ? bodyA->GetInverseMass() + ja * ja * bodyA->GetInverseInertia() : 0.0f) +
                (bodyB ? bodyB->GetInverseMass() + jb * jb * bodyB->GetInverseInertia() : 0.0f);
        }

        float SolveAxisVelocity(RigidBody2D* bodyA, RigidBody2D* bodyB,
            const Vector2& pointA, const Vector2& pointB, const Vector2& axis,
            float velocityBias, float maxImpulse)
        {
            const float inverseEffectiveMass = AxisMass(bodyA, bodyB, pointA, pointB, axis);
            if (inverseEffectiveMass <= Epsilon) return 0.0f;
            const float relativeVelocity = (PointVelocity(bodyB, pointB) - PointVelocity(bodyA, pointA)).Dot(axis);
            const float impulse = Clamp(-(relativeVelocity + velocityBias) / inverseEffectiveMass,
                -std::max(0.0f, maxImpulse), std::max(0.0f, maxImpulse));
            ApplyPairImpulse(bodyA, bodyB, impulse * axis, pointA, pointB);
            return impulse;
        }

        float SolveAngularVelocity(RigidBody2D* bodyA, RigidBody2D* bodyB,
            float velocityBias, float maxImpulse)
        {
            const float inverseEffectiveMass =
                (bodyA ? bodyA->GetInverseInertia() : 0.0f) +
                (bodyB ? bodyB->GetInverseInertia() : 0.0f);
            if (inverseEffectiveMass <= Epsilon) return 0.0f;
            const float relativeVelocity =
                (bodyB ? bodyB->GetAngularVelocity() : 0.0f) -
                (bodyA ? bodyA->GetAngularVelocity() : 0.0f);
            const float impulse = Clamp(-(relativeVelocity + velocityBias) / inverseEffectiveMass,
                -std::max(0.0f, maxImpulse), std::max(0.0f, maxImpulse));
            ApplyPairAngularImpulse(bodyA, bodyB, impulse);
            return impulse;
        }

        void ApplyPositionImpulse(RigidBody2D* bodyA, RigidBody2D* bodyB,
            const Vector2& impulse, const Vector2& pointA, const Vector2& pointB)
        {
            if (IsDynamic(bodyA))
            {
                Transform2 pose = bodyA->GetTransform();
                pose.Position -= bodyA->GetInverseMass() * impulse;
                pose.Angle -= bodyA->GetInverseInertia() * (pointA - pose.Position).Cross(impulse);
                bodyA->SetTransform(pose);
            }
            if (IsDynamic(bodyB))
            {
                Transform2 pose = bodyB->GetTransform();
                pose.Position += bodyB->GetInverseMass() * impulse;
                pose.Angle += bodyB->GetInverseInertia() * (pointB - pose.Position).Cross(impulse);
                bodyB->SetTransform(pose);
            }
        }

        void ApplyAngularPositionImpulse(RigidBody2D* bodyA, RigidBody2D* bodyB, float impulse)
        {
            if (IsDynamic(bodyA))
            {
                Transform2 pose = bodyA->GetTransform();
                pose.Angle -= bodyA->GetInverseInertia() * impulse;
                bodyA->SetTransform(pose);
            }
            if (IsDynamic(bodyB))
            {
                Transform2 pose = bodyB->GetTransform();
                pose.Angle += bodyB->GetInverseInertia() * impulse;
                bodyB->SetTransform(pose);
            }
        }

        bool SolveAxisPosition(RigidBody2D* bodyA, RigidBody2D* bodyB,
            const Vector2& pointA, const Vector2& pointB, const Vector2& axis,
            float error, const ConstraintSettings2D& settings)
        {
            const float softError = SoftError(error, std::max(0.0f, settings.PositionSlop));
            if (softError == 0.0f) return true;
            const float inverseEffectiveMass = AxisMass(bodyA, bodyB, pointA, pointB, axis);
            if (inverseEffectiveMass <= Epsilon) return false;
            const float correction = Clamp(softError, -settings.MaxPositionCorrection,
                settings.MaxPositionCorrection);
            ApplyPositionImpulse(bodyA, bodyB, (-correction / inverseEffectiveMass) * axis, pointA, pointB);
            return false;
        }

        bool SolveAnglePosition(RigidBody2D* bodyA, RigidBody2D* bodyB,
            float error, const ConstraintSettings2D& settings)
        {
            const float softError = SoftError(WrapAngle(error), std::max(0.0f, settings.PositionSlop));
            if (softError == 0.0f) return true;
            const float inverseEffectiveMass =
                (bodyA ? bodyA->GetInverseInertia() : 0.0f) +
                (bodyB ? bodyB->GetInverseInertia() : 0.0f);
            if (inverseEffectiveMass <= Epsilon) return false;
            const float correction = Clamp(softError, -settings.MaxPositionCorrection,
                settings.MaxPositionCorrection);
            ApplyAngularPositionImpulse(bodyA, bodyB, -correction / inverseEffectiveMass);
            return false;
        }

        bool ValidPair(const RigidBody2D* bodyA, const RigidBody2D* bodyB)
        {
            return bodyA && bodyB && bodyA != bodyB;
        }
    }

    DistanceConstraint2D::DistanceConstraint2D(const DistanceConstraintDesc2D& description)
        : Description(description)
    {
        if (ValidPair(Description.BodyA, Description.BodyB))
        {
            const float currentLength = (WorldAnchor(Description.BodyB, Description.LocalAnchorB) -
                WorldAnchor(Description.BodyA, Description.LocalAnchorA)).Length();
            RestLength = Description.Length >= 0.0f ? Description.Length : currentLength;
        }
    }

    void DistanceConstraint2D::SolveVelocity(float dt)
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return;
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 delta = pointB - pointA;
        const float length = delta.Length();
        if (length <= Epsilon) return;
        const float step = SafeStep(dt);
        const float bias = Description.Settings.VelocityBiasFactor * (length - RestLength) / step;
        SolveAxisVelocity(Description.BodyA, Description.BodyB, pointA, pointB, delta / length,
            bias, Description.Settings.MaxForce * step);
    }

    bool DistanceConstraint2D::SolvePosition()
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return true;
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 delta = pointB - pointA;
        const float length = delta.Length();
        if (length <= Epsilon) return RestLength <= Description.Settings.PositionSlop;
        return SolveAxisPosition(Description.BodyA, Description.BodyB, pointA, pointB,
            delta / length, length - RestLength, Description.Settings);
    }

    RevoluteConstraint2D::RevoluteConstraint2D(const RevoluteConstraintDesc2D& description)
        : Description(description)
    {
        if (ValidPair(Description.BodyA, Description.BodyB))
            ReferenceAngle = WrapAngle(Description.BodyB->GetTransform().Angle - Description.BodyA->GetTransform().Angle);
        if (Description.LowerAngle > Description.UpperAngle)
            std::swap(Description.LowerAngle, Description.UpperAngle);
    }

    void RevoluteConstraint2D::SolveVelocity(float dt)
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return;
        const float step = SafeStep(dt);
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 error = pointB - pointA;
        const float maxImpulse = Description.Settings.MaxForce * step;
        SolveAxisVelocity(Description.BodyA, Description.BodyB, pointA, pointB, Vector2::UnitX(),
            Description.Settings.VelocityBiasFactor * error.X / step, maxImpulse);
        SolveAxisVelocity(Description.BodyA, Description.BodyB, pointA, pointB, Vector2::UnitY(),
            Description.Settings.VelocityBiasFactor * error.Y / step, maxImpulse);
        if (Description.EnableMotor)
            SolveAngularVelocity(Description.BodyA, Description.BodyB, -Description.MotorSpeed,
                Description.MaxMotorTorque * step);
        if (Description.EnableLimit)
        {
            const float angle = WrapAngle(Description.BodyB->GetTransform().Angle -
                Description.BodyA->GetTransform().Angle - ReferenceAngle);
            float limitError = 0.0f;
            if (angle < Description.LowerAngle) limitError = angle - Description.LowerAngle;
            else if (angle > Description.UpperAngle) limitError = angle - Description.UpperAngle;
            if (limitError != 0.0f)
                SolveAngularVelocity(Description.BodyA, Description.BodyB,
                    Description.Settings.VelocityBiasFactor * limitError / step,
                    Description.Settings.MaxForce * step);
        }
    }

    bool RevoluteConstraint2D::SolvePosition()
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return true;
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 error = pointB - pointA;
        const bool xSolved = SolveAxisPosition(Description.BodyA, Description.BodyB, pointA, pointB,
            Vector2::UnitX(), error.X, Description.Settings);
        const bool ySolved = SolveAxisPosition(Description.BodyA, Description.BodyB, pointA, pointB,
            Vector2::UnitY(), error.Y, Description.Settings);
        bool angleSolved = true;
        if (Description.EnableLimit)
        {
            const float angle = WrapAngle(Description.BodyB->GetTransform().Angle -
                Description.BodyA->GetTransform().Angle - ReferenceAngle);
            float limitError = 0.0f;
            if (angle < Description.LowerAngle) limitError = angle - Description.LowerAngle;
            else if (angle > Description.UpperAngle) limitError = angle - Description.UpperAngle;
            angleSolved = SolveAnglePosition(Description.BodyA, Description.BodyB, limitError, Description.Settings);
        }
        return xSolved && ySolved && angleSolved;
    }

    PrismaticConstraint2D::PrismaticConstraint2D(const PrismaticConstraintDesc2D& description)
        : Description(description)
    {
        Description.LocalAxisA = Description.LocalAxisA.SafeUnit();
        if (Description.LocalAxisA.SquareLength() <= Epsilon) Description.LocalAxisA = Vector2::UnitX();
        if (ValidPair(Description.BodyA, Description.BodyB))
            ReferenceAngle = WrapAngle(Description.BodyB->GetTransform().Angle - Description.BodyA->GetTransform().Angle);
        if (Description.LowerTranslation > Description.UpperTranslation)
            std::swap(Description.LowerTranslation, Description.UpperTranslation);
    }

    void PrismaticConstraint2D::SolveVelocity(float dt)
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return;
        const float step = SafeStep(dt);
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 axis = Description.BodyA->GetTransform().TransformVector(Description.LocalAxisA).SafeUnit();
        const Vector2 perpendicular = axis.PerpendicularCCW();
        const float perpendicularError = (pointB - pointA).Dot(perpendicular);
        SolveAxisVelocity(Description.BodyA, Description.BodyB, pointA, pointB, perpendicular,
            Description.Settings.VelocityBiasFactor * perpendicularError / step,
            Description.Settings.MaxForce * step);
        const float angularError = WrapAngle(Description.BodyB->GetTransform().Angle -
            Description.BodyA->GetTransform().Angle - ReferenceAngle);
        SolveAngularVelocity(Description.BodyA, Description.BodyB,
            Description.Settings.VelocityBiasFactor * angularError / step,
            Description.Settings.MaxForce * step);
        if (Description.EnableMotor)
            SolveAxisVelocity(Description.BodyA, Description.BodyB, pointA, pointB, axis,
                -Description.MotorSpeed, Description.MaxMotorForce * step);
        if (Description.EnableLimit)
        {
            const float translation = (pointB - pointA).Dot(axis);
            float limitError = 0.0f;
            if (translation < Description.LowerTranslation) limitError = translation - Description.LowerTranslation;
            else if (translation > Description.UpperTranslation) limitError = translation - Description.UpperTranslation;
            if (limitError != 0.0f)
                SolveAxisVelocity(Description.BodyA, Description.BodyB, pointA, pointB, axis,
                    Description.Settings.VelocityBiasFactor * limitError / step,
                    Description.Settings.MaxForce * step);
        }
    }

    bool PrismaticConstraint2D::SolvePosition()
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return true;
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 axis = Description.BodyA->GetTransform().TransformVector(Description.LocalAxisA).SafeUnit();
        const Vector2 perpendicular = axis.PerpendicularCCW();
        const bool lateralSolved = SolveAxisPosition(Description.BodyA, Description.BodyB, pointA, pointB,
            perpendicular, (pointB - pointA).Dot(perpendicular), Description.Settings);
        const bool angleSolved = SolveAnglePosition(Description.BodyA, Description.BodyB,
            Description.BodyB->GetTransform().Angle - Description.BodyA->GetTransform().Angle - ReferenceAngle,
            Description.Settings);
        bool limitSolved = true;
        if (Description.EnableLimit)
        {
            const float translation = (pointB - pointA).Dot(axis);
            float error = 0.0f;
            if (translation < Description.LowerTranslation) error = translation - Description.LowerTranslation;
            else if (translation > Description.UpperTranslation) error = translation - Description.UpperTranslation;
            limitSolved = SolveAxisPosition(Description.BodyA, Description.BodyB, pointA, pointB,
                axis, error, Description.Settings);
        }
        return lateralSolved && angleSolved && limitSolved;
    }

    WeldConstraint2D::WeldConstraint2D(const WeldConstraintDesc2D& description)
        : Description(description)
    {
        if (ValidPair(Description.BodyA, Description.BodyB))
            ReferenceAngle = WrapAngle(Description.BodyB->GetTransform().Angle - Description.BodyA->GetTransform().Angle);
    }

    void WeldConstraint2D::SolveVelocity(float dt)
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return;
        const float step = SafeStep(dt);
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 error = pointB - pointA;
        const float maxImpulse = Description.Settings.MaxForce * step;
        SolveAxisVelocity(Description.BodyA, Description.BodyB, pointA, pointB, Vector2::UnitX(),
            Description.Settings.VelocityBiasFactor * error.X / step, maxImpulse);
        SolveAxisVelocity(Description.BodyA, Description.BodyB, pointA, pointB, Vector2::UnitY(),
            Description.Settings.VelocityBiasFactor * error.Y / step, maxImpulse);
        const float angleError = WrapAngle(Description.BodyB->GetTransform().Angle -
            Description.BodyA->GetTransform().Angle - ReferenceAngle);
        SolveAngularVelocity(Description.BodyA, Description.BodyB,
            Description.Settings.VelocityBiasFactor * angleError / step, maxImpulse);
    }

    bool WeldConstraint2D::SolvePosition()
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return true;
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 error = pointB - pointA;
        const bool xSolved = SolveAxisPosition(Description.BodyA, Description.BodyB, pointA, pointB,
            Vector2::UnitX(), error.X, Description.Settings);
        const bool ySolved = SolveAxisPosition(Description.BodyA, Description.BodyB, pointA, pointB,
            Vector2::UnitY(), error.Y, Description.Settings);
        const bool angleSolved = SolveAnglePosition(Description.BodyA, Description.BodyB,
            Description.BodyB->GetTransform().Angle - Description.BodyA->GetTransform().Angle - ReferenceAngle,
            Description.Settings);
        return xSolved && ySolved && angleSolved;
    }

    WheelConstraint2D::WheelConstraint2D(const WheelConstraintDesc2D& description)
        : Description(description)
    {
        Description.LocalAxisA = Description.LocalAxisA.SafeUnit();
        if (Description.LocalAxisA.SquareLength() <= Epsilon) Description.LocalAxisA = Vector2::UnitY();
        if (ValidPair(Description.BodyA, Description.BodyB))
        {
            const Vector2 axis = Description.BodyA->GetTransform().TransformVector(Description.LocalAxisA);
            RestTranslation = (WorldAnchor(Description.BodyB, Description.LocalAnchorB) -
                WorldAnchor(Description.BodyA, Description.LocalAnchorA)).Dot(axis);
        }
    }

    void WheelConstraint2D::SolveVelocity(float dt)
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return;
        const float step = SafeStep(dt);
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 axis = Description.BodyA->GetTransform().TransformVector(Description.LocalAxisA).SafeUnit();
        const Vector2 perpendicular = axis.PerpendicularCCW();
        const float lateralError = (pointB - pointA).Dot(perpendicular);
        SolveAxisVelocity(Description.BodyA, Description.BodyB, pointA, pointB, perpendicular,
            Description.Settings.VelocityBiasFactor * lateralError / step,
            Description.Settings.MaxForce * step);

        const float inverseEffectiveMass = AxisMass(Description.BodyA, Description.BodyB, pointA, pointB, axis);
        if (inverseEffectiveMass > Epsilon && Description.SuspensionFrequency > 0.0f)
        {
            const float effectiveMass = 1.0f / inverseEffectiveMass;
            const float omega = 2.0f * Pi * Description.SuspensionFrequency;
            const float displacement = (pointB - pointA).Dot(axis) - RestTranslation;
            const float speed = (PointVelocity(Description.BodyB, pointB) -
                PointVelocity(Description.BodyA, pointA)).Dot(axis);
            const float accelerationTerm = omega * omega * displacement +
                2.0f * std::max(0.0f, Description.SuspensionDamping) * omega * speed;
            const float impulse = Clamp(-effectiveMass * accelerationTerm * step,
                -Description.Settings.MaxForce * step, Description.Settings.MaxForce * step);
            ApplyPairImpulse(Description.BodyA, Description.BodyB, impulse * axis, pointA, pointB);
        }
        if (Description.EnableMotor)
            SolveAngularVelocity(Description.BodyA, Description.BodyB, -Description.MotorSpeed,
                Description.MaxMotorTorque * step);
    }

    bool WheelConstraint2D::SolvePosition()
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return true;
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 axis = Description.BodyA->GetTransform().TransformVector(Description.LocalAxisA).SafeUnit();
        const Vector2 perpendicular = axis.PerpendicularCCW();
        return SolveAxisPosition(Description.BodyA, Description.BodyB, pointA, pointB, perpendicular,
            (pointB - pointA).Dot(perpendicular), Description.Settings);
    }

    MotorConstraint2D::MotorConstraint2D(const MotorConstraintDesc2D& description)
        : Description(description)
    {
    }

    void MotorConstraint2D::SolveVelocity(float dt)
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return;
        const float step = SafeStep(dt);
        const Vector2 desiredPosition = Description.BodyA->GetTransform().TransformPoint(Description.LinearOffset);
        const Vector2 actualPosition = Description.BodyB->GetTransform().Position;
        const Vector2 error = actualPosition - desiredPosition;
        const float maxLinearImpulse = Description.MaxForce * step;
        SolveAxisVelocity(Description.BodyA, Description.BodyB, desiredPosition, actualPosition, Vector2::UnitX(),
            Description.CorrectionFactor * error.X / step, maxLinearImpulse);
        SolveAxisVelocity(Description.BodyA, Description.BodyB, desiredPosition, actualPosition, Vector2::UnitY(),
            Description.CorrectionFactor * error.Y / step, maxLinearImpulse);
        const float angleError = WrapAngle(Description.BodyB->GetTransform().Angle -
            Description.BodyA->GetTransform().Angle - Description.AngularOffset);
        SolveAngularVelocity(Description.BodyA, Description.BodyB,
            Description.CorrectionFactor * angleError / step, Description.MaxTorque * step);
    }

    bool MotorConstraint2D::SolvePosition()
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return true;
        const Vector2 desiredPosition = Description.BodyA->GetTransform().TransformPoint(Description.LinearOffset);
        const Vector2 actualPosition = Description.BodyB->GetTransform().Position;
        const Vector2 error = actualPosition - desiredPosition;
        const bool xSolved = SolveAxisPosition(Description.BodyA, Description.BodyB, desiredPosition, actualPosition,
            Vector2::UnitX(), error.X * Description.CorrectionFactor, Description.Settings);
        const bool ySolved = SolveAxisPosition(Description.BodyA, Description.BodyB, desiredPosition, actualPosition,
            Vector2::UnitY(), error.Y * Description.CorrectionFactor, Description.Settings);
        const bool angleSolved = SolveAnglePosition(Description.BodyA, Description.BodyB,
            (Description.BodyB->GetTransform().Angle - Description.BodyA->GetTransform().Angle -
                Description.AngularOffset) * Description.CorrectionFactor, Description.Settings);
        return xSolved && ySolved && angleSolved;
    }

    MouseConstraint2D::MouseConstraint2D(const MouseConstraintDesc2D& description)
        : Description(description)
    {
    }

    void MouseConstraint2D::SolveVelocity(float dt)
    {
        if (!Description.Body || !IsDynamic(Description.Body)) return;
        const float step = SafeStep(dt);
        const Vector2 point = WorldAnchor(Description.Body, Description.LocalAnchor);
        const Vector2 error = point - Description.Target;
        const Vector2 velocity = PointVelocity(Description.Body, point);
        const float omega = 2.0f * Pi * std::max(0.0f, Description.Frequency);
        const float damping = 2.0f * std::max(0.0f, Description.Damping) * omega;
        const float maxImpulse = std::max(0.0f, Description.MaxForce) * step;
        const Vector2 desiredImpulse = -(omega * omega * error + damping * velocity) *
            (Description.Body->GetInverseMass() > Epsilon ? step / Description.Body->GetInverseMass() : 0.0f);
        Vector2 impulse = desiredImpulse;
        const float length = impulse.Length();
        if (length > maxImpulse && length > Epsilon) impulse *= maxImpulse / length;
        Description.Body->ApplyImpulse(impulse, point);
    }

    bool MouseConstraint2D::SolvePosition()
    {
        if (!Description.Body || !IsDynamic(Description.Body)) return true;
        const Vector2 point = WorldAnchor(Description.Body, Description.LocalAnchor);
        const Vector2 error = point - Description.Target;
        const bool xSolved = SolveAxisPosition(nullptr, Description.Body, Description.Target, point,
            Vector2::UnitX(), error.X, Description.Settings);
        const bool ySolved = SolveAxisPosition(nullptr, Description.Body, Description.Target, point,
            Vector2::UnitY(), error.Y, Description.Settings);
        return xSolved && ySolved;
    }

    PulleyConstraint2D::PulleyConstraint2D(const PulleyConstraintDesc2D& description)
        : Description(description)
    {
        Description.Ratio = std::max(0.01f, Description.Ratio);
        if (ValidPair(Description.BodyA, Description.BodyB))
        {
            const float lengthA = (WorldAnchor(Description.BodyA, Description.LocalAnchorA) -
                Description.GroundAnchorA).Length();
            const float lengthB = (WorldAnchor(Description.BodyB, Description.LocalAnchorB) -
                Description.GroundAnchorB).Length();
            ConstantLength = Description.TotalLength >= 0.0f ? Description.TotalLength :
                lengthA + Description.Ratio * lengthB;
        }
    }

    void PulleyConstraint2D::SolveVelocity(float dt)
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return;
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 deltaA = pointA - Description.GroundAnchorA;
        const Vector2 deltaB = pointB - Description.GroundAnchorB;
        const float lengthA = deltaA.Length();
        const float lengthB = deltaB.Length();
        if (lengthA <= Epsilon || lengthB <= Epsilon) return;
        const Vector2 axisA = deltaA / lengthA;
        const Vector2 axisB = deltaB / lengthB;
        const Vector2 ra = pointA - Description.BodyA->GetTransform().Position;
        const Vector2 rb = pointB - Description.BodyB->GetTransform().Position;
        const float ja = ra.Cross(axisA);
        const float jb = Description.Ratio * rb.Cross(axisB);
        const float k = Description.BodyA->GetInverseMass() + ja * ja * Description.BodyA->GetInverseInertia() +
            Description.Ratio * Description.Ratio * Description.BodyB->GetInverseMass() +
            jb * jb * Description.BodyB->GetInverseInertia();
        if (k <= Epsilon) return;
        const float step = SafeStep(dt);
        const float speed = PointVelocity(Description.BodyA, pointA).Dot(axisA) +
            Description.Ratio * PointVelocity(Description.BodyB, pointB).Dot(axisB);
        const float error = lengthA + Description.Ratio * lengthB - ConstantLength;
        const float impulse = Clamp(-(speed + Description.Settings.VelocityBiasFactor * error / step) / k,
            -Description.Settings.MaxForce * step, Description.Settings.MaxForce * step);
        Description.BodyA->ApplyImpulse(impulse * axisA, pointA);
        Description.BodyB->ApplyImpulse((Description.Ratio * impulse) * axisB, pointB);
    }

    bool PulleyConstraint2D::SolvePosition()
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return true;
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const Vector2 deltaA = pointA - Description.GroundAnchorA;
        const Vector2 deltaB = pointB - Description.GroundAnchorB;
        const float lengthA = deltaA.Length();
        const float lengthB = deltaB.Length();
        if (lengthA <= Epsilon || lengthB <= Epsilon) return false;
        const Vector2 axisA = deltaA / lengthA;
        const Vector2 axisB = deltaB / lengthB;
        const float error = lengthA + Description.Ratio * lengthB - ConstantLength;
        const float softError = SoftError(error, Description.Settings.PositionSlop);
        if (softError == 0.0f) return true;
        const Vector2 ra = pointA - Description.BodyA->GetTransform().Position;
        const Vector2 rb = pointB - Description.BodyB->GetTransform().Position;
        const float ja = ra.Cross(axisA);
        const float jb = Description.Ratio * rb.Cross(axisB);
        const float k = Description.BodyA->GetInverseMass() + ja * ja * Description.BodyA->GetInverseInertia() +
            Description.Ratio * Description.Ratio * Description.BodyB->GetInverseMass() +
            jb * jb * Description.BodyB->GetInverseInertia();
        if (k <= Epsilon) return false;
        const float impulse = -Clamp(softError, -Description.Settings.MaxPositionCorrection,
            Description.Settings.MaxPositionCorrection) / k;
        if (IsDynamic(Description.BodyA))
        {
            Transform2 pose = Description.BodyA->GetTransform();
            pose.Position += Description.BodyA->GetInverseMass() * impulse * axisA;
            pose.Angle += Description.BodyA->GetInverseInertia() * ja * impulse;
            Description.BodyA->SetTransform(pose);
        }
        if (IsDynamic(Description.BodyB))
        {
            Transform2 pose = Description.BodyB->GetTransform();
            pose.Position += Description.BodyB->GetInverseMass() * Description.Ratio * impulse * axisB;
            pose.Angle += Description.BodyB->GetInverseInertia() * jb * impulse;
            Description.BodyB->SetTransform(pose);
        }
        return false;
    }

    GearConstraint2D::GearConstraint2D(const GearConstraintDesc2D& description)
        : Description(description)
    {
        if (ValidPair(Description.BodyA, Description.BodyB))
            Constant = Description.AutoConstant ? Description.BodyA->GetTransform().Angle +
                Description.Ratio * Description.BodyB->GetTransform().Angle : Description.Constant;
    }

    void GearConstraint2D::SolveVelocity(float dt)
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return;
        const float inverseEffectiveMass = Description.BodyA->GetInverseInertia() +
            Description.Ratio * Description.Ratio * Description.BodyB->GetInverseInertia();
        if (inverseEffectiveMass <= Epsilon) return;
        const float step = SafeStep(dt);
        const float error = Description.BodyA->GetTransform().Angle +
            Description.Ratio * Description.BodyB->GetTransform().Angle - Constant;
        const float speed = Description.BodyA->GetAngularVelocity() +
            Description.Ratio * Description.BodyB->GetAngularVelocity();
        const float impulse = Clamp(-(speed + Description.Settings.VelocityBiasFactor * error / step) /
            inverseEffectiveMass, -Description.MaxTorque * step, Description.MaxTorque * step);
        Description.BodyA->ApplyAngularImpulse(impulse);
        Description.BodyB->ApplyAngularImpulse(Description.Ratio * impulse);
    }

    bool GearConstraint2D::SolvePosition()
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return true;
        const float error = Description.BodyA->GetTransform().Angle +
            Description.Ratio * Description.BodyB->GetTransform().Angle - Constant;
        const float softError = SoftError(error, Description.Settings.PositionSlop);
        if (softError == 0.0f) return true;
        const float k = Description.BodyA->GetInverseInertia() +
            Description.Ratio * Description.Ratio * Description.BodyB->GetInverseInertia();
        if (k <= Epsilon) return false;
        const float impulse = -Clamp(softError, -Description.Settings.MaxPositionCorrection,
            Description.Settings.MaxPositionCorrection) / k;
        if (IsDynamic(Description.BodyA))
        {
            Transform2 pose = Description.BodyA->GetTransform();
            pose.Angle += Description.BodyA->GetInverseInertia() * impulse;
            Description.BodyA->SetTransform(pose);
        }
        if (IsDynamic(Description.BodyB))
        {
            Transform2 pose = Description.BodyB->GetTransform();
            pose.Angle += Description.BodyB->GetInverseInertia() * Description.Ratio * impulse;
            Description.BodyB->SetTransform(pose);
        }
        return false;
    }

    FrictionConstraint2D::FrictionConstraint2D(const FrictionConstraintDesc2D& description)
        : Description(description)
    {
    }

    void FrictionConstraint2D::SolveVelocity(float dt)
    {
        if (!ValidPair(Description.BodyA, Description.BodyB)) return;
        const float step = SafeStep(dt);
        const Vector2 pointA = WorldAnchor(Description.BodyA, Description.LocalAnchorA);
        const Vector2 pointB = WorldAnchor(Description.BodyB, Description.LocalAnchorB);
        const float maxImpulse = std::max(0.0f, Description.MaxForce) * step;
        SolveAxisVelocity(Description.BodyA, Description.BodyB, pointA, pointB,
            Vector2::UnitX(), 0.0f, maxImpulse);
        SolveAxisVelocity(Description.BodyA, Description.BodyB, pointA, pointB,
            Vector2::UnitY(), 0.0f, maxImpulse);
        SolveAngularVelocity(Description.BodyA, Description.BodyB, 0.0f,
            std::max(0.0f, Description.MaxTorque) * step);
    }

    bool FrictionConstraint2D::SolvePosition()
    {
        return true;
    }
}
