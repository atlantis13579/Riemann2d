#pragma once

#include "Dynamics/RigidBody2D.h"

namespace Riemann2D
{
    struct ConstraintSettings2D
    {
        float MaxForce = 1000.0f;
        float PositionSlop = 0.005f;
        float MaxPositionCorrection = 0.2f;
        float VelocityBiasFactor = 0.2f;
    };

    class IConstraint2D
    {
    public:
        virtual ~IConstraint2D() = default;
        virtual void SolveVelocity(float dt) = 0;
        virtual bool SolvePosition() = 0;
    };

    struct DistanceConstraintDesc2D
    {
        RigidBody2D* BodyA = nullptr;
        RigidBody2D* BodyB = nullptr;
        Vector2 LocalAnchorA = Vector2::Zero();
        Vector2 LocalAnchorB = Vector2::Zero();
        float Length = -1.0f;
        ConstraintSettings2D Settings;
    };

    class DistanceConstraint2D final : public IConstraint2D
    {
    public:
        explicit DistanceConstraint2D(const DistanceConstraintDesc2D& description);
        void SolveVelocity(float dt) override;
        bool SolvePosition() override;

    private:
        DistanceConstraintDesc2D Description;
        float RestLength = 0.0f;
    };

    struct RevoluteConstraintDesc2D
    {
        RigidBody2D* BodyA = nullptr;
        RigidBody2D* BodyB = nullptr;
        Vector2 LocalAnchorA = Vector2::Zero();
        Vector2 LocalAnchorB = Vector2::Zero();
        bool EnableLimit = false;
        float LowerAngle = -Pi;
        float UpperAngle = Pi;
        bool EnableMotor = false;
        float MotorSpeed = 0.0f;
        float MaxMotorTorque = 100.0f;
        ConstraintSettings2D Settings;
    };

    class RevoluteConstraint2D final : public IConstraint2D
    {
    public:
        explicit RevoluteConstraint2D(const RevoluteConstraintDesc2D& description);
        void SolveVelocity(float dt) override;
        bool SolvePosition() override;
        void SetMotorSpeed(float speed) { Description.MotorSpeed = speed; }

    private:
        RevoluteConstraintDesc2D Description;
        float ReferenceAngle = 0.0f;
    };

    struct PrismaticConstraintDesc2D
    {
        RigidBody2D* BodyA = nullptr;
        RigidBody2D* BodyB = nullptr;
        Vector2 LocalAnchorA = Vector2::Zero();
        Vector2 LocalAnchorB = Vector2::Zero();
        Vector2 LocalAxisA = Vector2::UnitX();
        bool EnableLimit = false;
        float LowerTranslation = -1.0f;
        float UpperTranslation = 1.0f;
        bool EnableMotor = false;
        float MotorSpeed = 0.0f;
        float MaxMotorForce = 100.0f;
        ConstraintSettings2D Settings;
    };

    class PrismaticConstraint2D final : public IConstraint2D
    {
    public:
        explicit PrismaticConstraint2D(const PrismaticConstraintDesc2D& description);
        void SolveVelocity(float dt) override;
        bool SolvePosition() override;
        void SetMotorSpeed(float speed) { Description.MotorSpeed = speed; }

    private:
        PrismaticConstraintDesc2D Description;
        float ReferenceAngle = 0.0f;
    };

    struct WeldConstraintDesc2D
    {
        RigidBody2D* BodyA = nullptr;
        RigidBody2D* BodyB = nullptr;
        Vector2 LocalAnchorA = Vector2::Zero();
        Vector2 LocalAnchorB = Vector2::Zero();
        ConstraintSettings2D Settings;
    };

    class WeldConstraint2D final : public IConstraint2D
    {
    public:
        explicit WeldConstraint2D(const WeldConstraintDesc2D& description);
        void SolveVelocity(float dt) override;
        bool SolvePosition() override;

    private:
        WeldConstraintDesc2D Description;
        float ReferenceAngle = 0.0f;
    };

    struct WheelConstraintDesc2D
    {
        RigidBody2D* BodyA = nullptr;
        RigidBody2D* BodyB = nullptr;
        Vector2 LocalAnchorA = Vector2::Zero();
        Vector2 LocalAnchorB = Vector2::Zero();
        Vector2 LocalAxisA = Vector2::UnitY();
        float SuspensionFrequency = 5.0f;
        float SuspensionDamping = 0.7f;
        bool EnableMotor = false;
        float MotorSpeed = 0.0f;
        float MaxMotorTorque = 100.0f;
        ConstraintSettings2D Settings;
    };

    class WheelConstraint2D final : public IConstraint2D
    {
    public:
        explicit WheelConstraint2D(const WheelConstraintDesc2D& description);
        void SolveVelocity(float dt) override;
        bool SolvePosition() override;
        void SetMotorSpeed(float speed) { Description.MotorSpeed = speed; }

    private:
        WheelConstraintDesc2D Description;
        float RestTranslation = 0.0f;
    };

    struct MotorConstraintDesc2D
    {
        RigidBody2D* BodyA = nullptr;
        RigidBody2D* BodyB = nullptr;
        Vector2 LinearOffset = Vector2::Zero();
        float AngularOffset = 0.0f;
        float CorrectionFactor = 0.3f;
        float MaxForce = 1000.0f;
        float MaxTorque = 100.0f;
        ConstraintSettings2D Settings;
    };

    class MotorConstraint2D final : public IConstraint2D
    {
    public:
        explicit MotorConstraint2D(const MotorConstraintDesc2D& description);
        void SolveVelocity(float dt) override;
        bool SolvePosition() override;
        void SetLinearOffset(const Vector2& offset) { Description.LinearOffset = offset; }
        void SetAngularOffset(float offset) { Description.AngularOffset = offset; }

    private:
        MotorConstraintDesc2D Description;
    };

    struct MouseConstraintDesc2D
    {
        RigidBody2D* Body = nullptr;
        Vector2 LocalAnchor = Vector2::Zero();
        Vector2 Target = Vector2::Zero();
        float Frequency = 5.0f;
        float Damping = 0.8f;
        float MaxForce = 1000.0f;
        ConstraintSettings2D Settings;
    };

    class MouseConstraint2D final : public IConstraint2D
    {
    public:
        explicit MouseConstraint2D(const MouseConstraintDesc2D& description);
        void SolveVelocity(float dt) override;
        bool SolvePosition() override;
        void SetTarget(const Vector2& target) { Description.Target = target; }

    private:
        MouseConstraintDesc2D Description;
    };

    struct PulleyConstraintDesc2D
    {
        RigidBody2D* BodyA = nullptr;
        RigidBody2D* BodyB = nullptr;
        Vector2 LocalAnchorA = Vector2::Zero();
        Vector2 LocalAnchorB = Vector2::Zero();
        Vector2 GroundAnchorA = {-1.0f, 1.0f};
        Vector2 GroundAnchorB = {1.0f, 1.0f};
        float Ratio = 1.0f;
        float TotalLength = -1.0f;
        ConstraintSettings2D Settings;
    };

    class PulleyConstraint2D final : public IConstraint2D
    {
    public:
        explicit PulleyConstraint2D(const PulleyConstraintDesc2D& description);
        void SolveVelocity(float dt) override;
        bool SolvePosition() override;

    private:
        PulleyConstraintDesc2D Description;
        float ConstantLength = 0.0f;
    };

    struct GearConstraintDesc2D
    {
        RigidBody2D* BodyA = nullptr;
        RigidBody2D* BodyB = nullptr;
        float Ratio = 1.0f;
        float Constant = 0.0f;
        bool AutoConstant = true;
        float MaxTorque = 1000.0f;
        ConstraintSettings2D Settings;
    };

    class GearConstraint2D final : public IConstraint2D
    {
    public:
        explicit GearConstraint2D(const GearConstraintDesc2D& description);
        void SolveVelocity(float dt) override;
        bool SolvePosition() override;

    private:
        GearConstraintDesc2D Description;
        float Constant = 0.0f;
    };

    struct FrictionConstraintDesc2D
    {
        RigidBody2D* BodyA = nullptr;
        RigidBody2D* BodyB = nullptr;
        Vector2 LocalAnchorA = Vector2::Zero();
        Vector2 LocalAnchorB = Vector2::Zero();
        float MaxForce = 100.0f;
        float MaxTorque = 10.0f;
    };

    class FrictionConstraint2D final : public IConstraint2D
    {
    public:
        explicit FrictionConstraint2D(const FrictionConstraintDesc2D& description);
        void SolveVelocity(float dt) override;
        bool SolvePosition() override;

    private:
        FrictionConstraintDesc2D Description;
    };
}
