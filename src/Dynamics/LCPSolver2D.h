#pragma once

#include "Collision/Contact2.h"
#include "Dynamics/LinearAlgebraBackend2D.h"

#include <vector>

namespace Riemann2D
{
    class LCPSolver2D
    {
    public:
        void Solve(std::vector<Contact2>& contacts, SolverBackend2D backend);
        float GetLastComplementarityResidual() const { return LastResidual; }

    private:
        float LastResidual = 0.0f;
    };
}
