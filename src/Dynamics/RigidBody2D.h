#pragma once

#include "Collision/Shape2.h"

#include <memory>

namespace Riemann2D
{
    enum class BodyType2D
    {
        Static,
        Kinematic,
        Dynamic
    };

    struct RigidBodyDesc2D
    {
        BodyType2D Type = BodyType2D::Dynamic;
        std::shared_ptr<const Shape2> Shape;
        Transform2 Transform;
        Vector2 LinearVelocity = Vector2::Zero();
        float AngularVelocity = 0.0f;
        float Density = 1.0f;
        float Friction = 0.5f;
        float Restitution = 0.05f;
        float LinearDamping = 0.02f;
        float AngularDamping = 0.02f;
        float GravityScale = 1.0f;
    };

    class RigidBody2D
    {
    public:
        explicit RigidBody2D(const RigidBodyDesc2D& description);

        BodyType2D GetType() const { return Type; }
        const Shape2& GetShape() const { return *Shape; }
        const Transform2& GetTransform() const { return Pose; }
        Vector2 GetLinearVelocity() const { return LinearVelocity; }
        float GetAngularVelocity() const { return AngularVelocity; }
        float GetInverseMass() const { return InverseMass; }
        float GetInverseInertia() const { return InverseInertia; }
        float GetFriction() const { return Friction; }
        float GetRestitution() const { return Restitution; }

        void SetTransform(const Transform2& transform) { Pose = transform; }
        void SetLinearVelocity(const Vector2& velocity) { LinearVelocity = velocity; }
        void SetAngularVelocity(float velocity) { AngularVelocity = velocity; }
        void ApplyForce(const Vector2& force) { Force += force; }
        void ApplyForceAtPoint(const Vector2& force, const Vector2& worldPoint)
        {
            if (Type != BodyType2D::Dynamic) return;
            Force += force;
            Torque += (worldPoint - Pose.Position).Cross(force);
        }
        void ApplyTorque(float torque) { Torque += torque; }
        void ApplyImpulse(const Vector2& impulse, const Vector2& worldPoint);
        void ApplyAngularImpulse(float impulse)
        {
            if (Type == BodyType2D::Dynamic) AngularVelocity += impulse * InverseInertia;
        }

    private:
        friend class PhysicsWorld2D;

        BodyType2D Type;
        std::shared_ptr<const Shape2> Shape;
        Transform2 Pose;
        Vector2 LinearVelocity;
        float AngularVelocity;
        Vector2 Force = Vector2::Zero();
        float Torque = 0.0f;
        float InverseMass = 0.0f;
        float InverseInertia = 0.0f;
        float Friction;
        float Restitution;
        float LinearDamping;
        float AngularDamping;
        float GravityScale;
    };
}
