#pragma once

#include "Maths/Vector2.h"

#include <cstdint>

namespace Riemann2D
{
    class RigidBody2D;

    struct Contact2
    {
        RigidBody2D* BodyA = nullptr;
        RigidBody2D* BodyB = nullptr;
        Vector2 Normal = Vector2::UnitY();
        Vector2 Point = Vector2::Zero();
        float Penetration = 0.0f;
        float NormalImpulse = 0.0f;
        float TangentImpulse = 0.0f;
        std::int32_t TriangleIndexA = -1;
        std::int32_t TriangleIndexB = -1;
    };

    bool GenerateContact(RigidBody2D& bodyA, RigidBody2D& bodyB, Contact2& contact);
}
