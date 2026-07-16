#include "RigidBody2D.h"


namespace Riemann2D
{
    RigidBody2D::RigidBody2D(const RigidBodyDesc2D& description)
        : Type(description.Type)
        , Shape(description.Shape)
        , Pose(description.Transform)
        , LinearVelocity(description.LinearVelocity)
        , AngularVelocity(description.AngularVelocity)
        , Friction(description.Friction)
        , Restitution(description.Restitution)
        , LinearDamping(description.LinearDamping)
        , AngularDamping(description.AngularDamping)
        , GravityScale(description.GravityScale)
    {
        if (Type == BodyType2D::Dynamic)
        {
            const MassProperties2 mass = Shape->ComputeMass(description.Density);
            InverseMass = 1.0f / mass.Mass;
            InverseInertia = 1.0f / mass.Inertia;
        }
    }

    void RigidBody2D::ApplyImpulse(const Vector2& impulse, const Vector2& worldPoint)
    {
        if (Type != BodyType2D::Dynamic)
        {
            return;
        }
        LinearVelocity += impulse * InverseMass;
        AngularVelocity += (worldPoint - Pose.Position).Cross(impulse) * InverseInertia;
    }
}
