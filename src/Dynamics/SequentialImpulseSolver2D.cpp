#include "SequentialImpulseSolver2D.h"

#include "Dynamics/ContactSolverUtilities2D.h"

#include <algorithm>
#include <vector>

namespace Riemann2D
{
    void SequentialImpulseSolver2D::Solve(std::vector<Contact2>& contacts, int iterations,
                                          SolverBackend2D backend) const
    {
        (void)backend;
        if (iterations <= 0 || contacts.empty())
        {
            return;
        }

        std::vector<float> targetNormalVelocities(contacts.size(), 0.0f);
        for (std::size_t index = 0; index < contacts.size(); ++index)
        {
            Contact2& contact = contacts[index];
            contact.NormalImpulse = 0.0f;
            contact.TangentImpulse = 0.0f;
            const Vector2 relativeVelocity =
                SolverDetail::VelocityAtPoint(*contact.BodyB, contact.Point) -
                SolverDetail::VelocityAtPoint(*contact.BodyA, contact.Point);
            const float normalVelocity = relativeVelocity.Dot(contact.Normal);
            if (normalVelocity < -1.0f)
            {
                targetNormalVelocities[index] =
                    -std::max(contact.BodyA->GetRestitution(), contact.BodyB->GetRestitution()) * normalVelocity;
            }
        }

        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            for (std::size_t index = 0; index < contacts.size(); ++index)
            {
                Contact2& contact = contacts[index];
                const Vector2 relativeVelocity =
                    SolverDetail::VelocityAtPoint(*contact.BodyB, contact.Point) -
                    SolverDetail::VelocityAtPoint(*contact.BodyA, contact.Point);
                const float normalVelocity = relativeVelocity.Dot(contact.Normal);
                const float normalMass = SolverDetail::EffectiveMass(contact, contact.Normal);
                const float oldNormalImpulse = contact.NormalImpulse;
                contact.NormalImpulse = std::max(oldNormalImpulse +
                    normalMass * (targetNormalVelocities[index] - normalVelocity), 0.0f);
                SolverDetail::ApplyImpulse(contact,
                    (contact.NormalImpulse - oldNormalImpulse) * contact.Normal);

                const Vector2 tangent = SolverDetail::Tangent(contact.Normal);
                const Vector2 velocityAfterNormal =
                    SolverDetail::VelocityAtPoint(*contact.BodyB, contact.Point) -
                    SolverDetail::VelocityAtPoint(*contact.BodyA, contact.Point);
                const float tangentMass = SolverDetail::EffectiveMass(contact, tangent);
                const float maximumFriction = SolverDetail::MixedFriction(contact) * contact.NormalImpulse;
                const float oldTangentImpulse = contact.TangentImpulse;
                contact.TangentImpulse = Clamp(oldTangentImpulse -
                    tangentMass * velocityAfterNormal.Dot(tangent), -maximumFriction, maximumFriction);
                SolverDetail::ApplyImpulse(contact,
                    (contact.TangentImpulse - oldTangentImpulse) * tangent);
            }
        }
    }
}
