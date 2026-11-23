#include "LinearAlgebraBackend2D.h"

#include <algorithm>
#include <cmath>

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#define RIEMANN2D_LINEAR_ALGEBRA_SSE2 1
#include <emmintrin.h>
#elif defined(_M_ARM64) || defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#define RIEMANN2D_LINEAR_ALGEBRA_NEON 1
#include <arm_neon.h>
#endif

namespace Riemann2D
{
    namespace
    {
        float DotScalar(const float* left, const float* right, int n)
        {
            float result = 0.0f;
            for (int index = 0; index < n; ++index)
            {
                result += left[index] * right[index];
            }
            return result;
        }

#if defined(RIEMANN2D_LINEAR_ALGEBRA_SSE2)
        float DotSIMD(const float* left, const float* right, int n)
        {
            __m128 sum = _mm_setzero_ps();
            int index = 0;
            for (; index + 4 <= n; index += 4)
            {
                const __m128 leftValues = _mm_loadu_ps(left + index);
                const __m128 rightValues = _mm_loadu_ps(right + index);
                sum = _mm_add_ps(sum, _mm_mul_ps(leftValues, rightValues));
            }

            alignas(16) float lanes[4];
            _mm_store_ps(lanes, sum);
            float result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
            for (; index < n; ++index)
            {
                result += left[index] * right[index];
            }
            return result;
        }
#elif defined(RIEMANN2D_LINEAR_ALGEBRA_NEON)
        float DotSIMD(const float* left, const float* right, int n)
        {
            float32x4_t sum = vdupq_n_f32(0.0f);
            int index = 0;
            for (; index + 4 <= n; index += 4)
            {
                const float32x4_t leftValues = vld1q_f32(left + index);
                const float32x4_t rightValues = vld1q_f32(right + index);
                sum = vaddq_f32(sum, vmulq_f32(leftValues, rightValues));
            }

            float lanes[4];
            vst1q_f32(lanes, sum);
            float result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
            for (; index < n; ++index)
            {
                result += left[index] * right[index];
            }
            return result;
        }
#else
        float DotSIMD(const float* left, const float* right, int n)
        {
            return DotScalar(left, right, n);
        }
#endif
    }

    float Dot(const float* left, const float* right, int n, SolverBackend2D backend)
    {
        if (left == nullptr || right == nullptr || n <= 0)
        {
            return 0.0f;
        }

        return backend == SolverBackend2D::SIMDCPU
            ? DotSIMD(left, right, n)
            : DotScalar(left, right, n);
    }

    void DenseMatrixVectorMultiply(const float* matrix, const float* vector, float* output,
                                   int n, SolverBackend2D backend)
    {
        if (matrix == nullptr || vector == nullptr || output == nullptr || n <= 0)
        {
            return;
        }

        for (int row = 0; row < n; ++row)
        {
            output[row] = Dot(matrix + row * n, vector, n, backend);
        }
    }

    float MaximumAbsoluteResidual(const float* matrix, const float* solution,
                                  const float* rightHandSide, int n,
                                  SolverBackend2D backend)
    {
        if (matrix == nullptr || solution == nullptr || rightHandSide == nullptr || n <= 0)
        {
            return 0.0f;
        }

        float maximumResidual = 0.0f;
        for (int row = 0; row < n; ++row)
        {
            const float product = Dot(matrix + row * n, solution, n, backend);
            maximumResidual = std::max(maximumResidual,
                                       std::abs(product - rightHandSide[row]));
        }
        return maximumResidual;
    }
}
