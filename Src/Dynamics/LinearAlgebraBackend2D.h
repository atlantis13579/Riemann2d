#pragma once

namespace Riemann2D
{
    enum class SolverBackend2D
    {
        ScalarCPU,
        SIMDCPU
    };

    bool IsSolverBackendAvailable(SolverBackend2D backend);

    // Multiplies an n-by-n row-major matrix by a vector.
    void DenseMatrixVectorMultiply(const float* matrix, const float* vector, float* output,
                                   int n, SolverBackend2D backend);

    float Dot(const float* left, const float* right, int n, SolverBackend2D backend);

    // Returns max(abs(matrix * solution - rightHandSide)).
    float MaximumAbsoluteResidual(const float* matrix, const float* solution,
                                  const float* rightHandSide, int n,
                                  SolverBackend2D backend);
}
