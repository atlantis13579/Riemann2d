#pragma once

namespace Riemann2D
{
    class RigidBody2D;

    class ForceField2D
    {
    public:
        virtual ~ForceField2D() = default;
        virtual void Update(float deltaTime) { (void)deltaTime; }
        virtual bool ApplyForce(RigidBody2D& body) = 0;
    };
}
