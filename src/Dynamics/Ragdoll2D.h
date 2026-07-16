#pragma once

#include "Dynamics/PhysicsWorld2D.h"

#include <string>
#include <vector>

namespace Riemann2D
{
    struct RagdollPart2D
    {
        std::string Name;
        RigidBody2D* Body = nullptr;
    };

    class Ragdoll2D
    {
    public:
        bool Build(PhysicsWorld2D& world, const Vector2& origin, float scale = 1.0f);
        void Clear() { Parts.clear(); }
        const std::vector<RagdollPart2D>& GetParts() const { return Parts; }

    private:
        std::vector<RagdollPart2D> Parts;
    };
}
