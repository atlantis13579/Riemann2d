#pragma once

#include "Collision/Contact2.h"
#include "Dynamics/LinearAlgebraBackend2D.h"

#include <vector>

namespace Riemann2D
{
    class SequentialImpulseSolver2D
    {
    public:
        void Solve(std::vector<Contact2>& contacts, int iterations, SolverBackend2D backend) const;
    };
}
