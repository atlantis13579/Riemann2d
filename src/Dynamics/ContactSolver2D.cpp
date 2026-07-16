#include "ContactSolver2D.h"

namespace Riemann2D
{
    void ContactSolver2D::Solve(std::vector<Contact2>& contacts, ContactSolverType2D type,
                                SolverBackend2D backend, int iterations)
    {
        if (type == ContactSolverType2D::LCP)
        {
            LCP.Solve(contacts, backend);
            LastResidual = LCP.GetLastComplementarityResidual();
        }
        else
        {
            Sequential.Solve(contacts, iterations, backend);
            LastResidual = 0.0f;
        }
    }
}
