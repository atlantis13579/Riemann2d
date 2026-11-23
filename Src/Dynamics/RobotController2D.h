#pragma once

#include "Collision/Contact2.h"
#include "Dynamics/RigidBody2D.h"

#include <vector>

namespace Riemann2D
{
    struct RobotControllerSettings2D
    {
        Vector2 HalfExtent{0.35f, 0.9f};
        float SlopeLimitCosine = 0.70710678f;
    };

    // An upright robot in the XY plane. The world solver supplies floor,
    // wall and ceiling contacts, including contacts with other moving bodies.
    class RobotController2D
    {
    public:
        RobotController2D(const RobotController2D&) = delete;
        RobotController2D& operator=(const RobotController2D&) = delete;
        const Vector2& GetPosition() const { return Body.GetTransform().Position; }
        bool IsGrounded() const { return Grounded; }
        void SetMoveVelocity(float horizontalSpeed);
        bool Jump(float speed);

    private:
        friend class PhysicsWorld2D;
        RobotController2D(RigidBody2D& body, float slopeLimitCosine);
        void BeginStep(float gravityHalfStep);
        void FinishStep(float gravityHalfStep, const std::vector<Contact2>& contacts);

        RigidBody2D& Body;
        float SlopeLimitCosine;
        float WalkSpeed = 0.0f;
        bool Grounded = false;
    };
}
