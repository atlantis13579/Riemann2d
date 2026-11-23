#pragma once

#include "Collision/Contact2.h"
#include "Dynamics/RigidBody2D.h"
#include "Maths/Maths.h"

#include <algorithm>
#include <cmath>

namespace Riemann2D::SolverDetail
{
    inline Vector2 VelocityAtPoint(const RigidBody2D& body, const Vector2& point)
    {
        return body.GetLinearVelocity() + Cross(body.GetAngularVelocity(), point - body.GetTransform().Position);
    }

    inline float EffectiveMass(const Contact2& contact, const Vector2& axis)
    {
        const RigidBody2D& bodyA = *contact.BodyA;
        const RigidBody2D& bodyB = *contact.BodyB;
        const Vector2 radiusA = contact.Point - bodyA.GetTransform().Position;
        const Vector2 radiusB = contact.Point - bodyB.GetTransform().Position;
        const float angularA = radiusA.Cross(axis);
        const float angularB = radiusB.Cross(axis);
        const float inverseMass = bodyA.GetInverseMass() + bodyB.GetInverseMass() +
            angularA * angularA * bodyA.GetInverseInertia() +
            angularB * angularB * bodyB.GetInverseInertia();
        return inverseMass > Epsilon ? 1.0f / inverseMass : 0.0f;
    }

    inline void ApplyImpulse(Contact2& contact, const Vector2& impulse)
    {
        contact.BodyA->ApplyImpulse(-impulse, contact.Point);
        contact.BodyB->ApplyImpulse(impulse, contact.Point);
    }

    inline Vector2 Tangent(const Vector2& normal)
    {
        return normal.PerpendicularCCW();
    }

    inline float MixedFriction(const Contact2& contact)
    {
        return std::sqrt(contact.BodyA->GetFriction() * contact.BodyB->GetFriction());
    }
}
