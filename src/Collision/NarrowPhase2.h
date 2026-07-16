#pragma once

#include "Collision/SAT2.h"
#include "Collision/Contact2.h"

#include <memory>
#include <vector>

namespace Riemann2D
{
    class RigidBody2D;

    class NarrowPhase2
    {
    public:
        void GenerateContacts(const std::vector<std::unique_ptr<RigidBody2D>>& bodies,
                              const std::vector<BroadPhasePair2>& pairs,
                              std::vector<Contact2>& contacts) const;
    };
}
