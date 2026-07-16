#include "NarrowPhase2.h"

#include "Dynamics/RigidBody2D.h"

namespace Riemann2D
{
    void NarrowPhase2::GenerateContacts(
        const std::vector<std::unique_ptr<RigidBody2D>>& bodies,
        const std::vector<BroadPhasePair2>& pairs,
        std::vector<Contact2>& contacts) const
    {
        contacts.clear();
        contacts.reserve(pairs.size());
        for (const BroadPhasePair2& pair : pairs)
        {
            if (pair.A >= bodies.size() || pair.B >= bodies.size() || pair.A == pair.B)
            {
                continue;
            }
            Contact2 contact;
            if (GenerateContact(*bodies[pair.A], *bodies[pair.B], contact))
            {
                contacts.push_back(contact);
            }
        }
    }
}
