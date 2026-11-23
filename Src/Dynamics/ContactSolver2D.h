#pragma once

#include "Dynamics/LCPSolver2D.h"
#include "Dynamics/SequentialImpulseSolver2D.h"

namespace Riemann2D
{
    enum class ContactSolverType2D
    {
        SequentialImpulse,
        LCP
    };

    class ContactSolver2D
    {
    public:
        void Solve(std::vector<Contact2>& contacts, ContactSolverType2D type,
                   SolverBackend2D backend, int iterations);
        float GetLastResidual() const { return LastResidual; }

    private:
        SequentialImpulseSolver2D Sequential;
        LCPSolver2D LCP;
        float LastResidual = 0.0f;
    };
}
