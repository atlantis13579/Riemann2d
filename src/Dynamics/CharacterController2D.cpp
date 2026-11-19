#include "CharacterController2D.h"

#include "Dynamics/ContactSolverUtilities2D.h"

#include <cmath>
#include <stdexcept>

namespace Riemann2D
{
    CharacterController2D::CharacterController2D(RigidBody2D& body, float slopeLimitCosine)
        : Body(body), SlopeLimitCosine(slopeLimitCosine)
    {
    }

    void CharacterController2D::SetMoveVelocity(float horizontalSpeed)
    {
        if (!std::isfinite(horizontalSpeed))
            throw std::invalid_argument("Character walk speed must be finite");
        WalkSpeed = horizontalSpeed;
    }

    bool CharacterController2D::Jump(float speed)
    {
        if (!Grounded || !std::isfinite(speed) || speed <= 0.0f) return false;
        Body.SetLinearVelocity({Body.GetLinearVelocity().X, speed});
        Grounded = false;
        return true;
    }

    void CharacterController2D::BeginStep(float gravityHalfStep)
    {
        // The drift uses the midpoint velocity, giving an exact free-flight arc.
        Body.SetLinearVelocity({WalkSpeed, Body.GetLinearVelocity().Y + gravityHalfStep});
    }

    void CharacterController2D::FinishStep(float gravityHalfStep, const std::vector<Contact2>& contacts)
    {
        Body.SetLinearVelocity(Body.GetLinearVelocity() + Vector2{0.0f, gravityHalfStep});
        Grounded = false;
        for (const Contact2& contact : contacts)
        {
            if (contact.BodyA != &Body && contact.BodyB != &Body) continue;
            const bool isA = contact.BodyA == &Body;
            const Vector2 normal = isA ? -contact.Normal : contact.Normal;
            if (normal.Y < SlopeLimitCosine) continue;
            const RigidBody2D& support = isA ? *contact.BodyB : *contact.BodyA;
            const Vector2 relative = Body.GetLinearVelocity() - SolverDetail::VelocityAtPoint(support, contact.Point);
            if (relative.Dot(normal) <= Epsilon)
            {
                Grounded = true;
                break;
            }
        }
    }
}
