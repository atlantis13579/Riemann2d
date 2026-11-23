#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define R2_MPM_HD __host__ __device__
#else
#define R2_MPM_HD
#endif

namespace Riemann2D::Continuum
{
    R2_MPM_HD inline float Limit(float value, float minimum, float maximum)
    {
        return fminf(fmaxf(value, minimum), maximum);
    }

    struct Matrix2
    {
        float A = 0, B = 0, C = 0, D = 0;
        R2_MPM_HD Matrix2() {}
        R2_MPM_HD Matrix2(float a, float b, float c, float d) : A(a), B(b), C(c), D(d) {}
        R2_MPM_HD static Matrix2 Identity() { return {1, 0, 0, 1}; }
        R2_MPM_HD Matrix2 operator+(const Matrix2& m) const { return {A + m.A, B + m.B, C + m.C, D + m.D}; }
        R2_MPM_HD Matrix2 operator-(const Matrix2& m) const { return {A - m.A, B - m.B, C - m.C, D - m.D}; }
        R2_MPM_HD Matrix2 operator*(float s) const { return {A * s, B * s, C * s, D * s}; }
        R2_MPM_HD Matrix2 operator*(const Matrix2& m) const
        {
            return {A * m.A + B * m.C, A * m.B + B * m.D, C * m.A + D * m.C, C * m.B + D * m.D};
        }
        R2_MPM_HD Matrix2 Transpose() const { return {A, C, B, D}; }
        R2_MPM_HD float Determinant() const { return A * D - B * C; }
        R2_MPM_HD float X(float x, float y) const { return A * x + B * y; }
        R2_MPM_HD float Y(float x, float y) const { return C * x + D * y; }
    };

    struct Svd2
    {
        Matrix2 U, V;
        float X, Y;
    };

    R2_MPM_HD inline Svd2 Decompose(const Matrix2& f)
    {
        const float a = f.A * f.A + f.C * f.C;
        const float b = f.A * f.B + f.C * f.D;
        const float c = f.B * f.B + f.D * f.D;
        const float angle = 0.5f * atan2f(2 * b, a - c);
        const float cosine = cosf(angle), sine = sinf(angle);
        Svd2 result;
        result.V = {cosine, -sine, sine, cosine};
        const float x0 = f.X(cosine, sine), y0 = f.Y(cosine, sine);
        const float x1 = f.X(-sine, cosine), y1 = f.Y(-sine, cosine);
        result.X = fmaxf(sqrtf(x0 * x0 + y0 * y0), 1.e-6f);
        result.Y = fmaxf(sqrtf(x1 * x1 + y1 * y1), 1.e-6f);
        const float ux = x0 / result.X, uy = y0 / result.X;
        // Proper rotations prevent numerical inversions from creating tensile sand.
        result.U = {ux, -uy, uy, ux};
        if (result.X <= 1.e-6f) result.U = Matrix2::Identity();
        return result;
    }

    struct Material2D
    {
        float YoungModulus = 40000;
        float PoissonRatio = 0.2f;
        float SnowFraction = 1;
        float Viscosity = 2;
        float Hardening = 8;
        float CompressionYield = 0.025f;
        float StretchYield = 0.0075f;
        float FrictionAngle = 35;
    };

    R2_MPM_HD inline void Lame(const Material2D& material, float plasticVolume, float& mu, float& lambda)
    {
        const float hardening = expf(Limit(material.Hardening * (1 - plasticVolume) * material.SnowFraction, -2.0f, 1.6f));
        mu = material.YoungModulus * hardening / (2 * (1 + material.PoissonRatio));
        lambda = material.YoungModulus * hardening * material.PoissonRatio /
            ((1 + material.PoissonRatio) * (1 - 2 * material.PoissonRatio));
    }

    // Snow: fixed-corotated elasticity. Sand: Hencky elasticity with no tensile yield.
    R2_MPM_HD inline Matrix2 KirchhoffStress(const Matrix2& f, const Matrix2& c, float plasticVolume, const Material2D& material)
    {
        float mu, lambda;
        Lame(material, plasticVolume, mu, lambda);
        const Svd2 svd = Decompose(f);
        const Matrix2 rotation = svd.U * svd.V.Transpose();
        const float j = fmaxf(f.Determinant(), 1.e-6f);
        const Matrix2 snow = (f - rotation) * f.Transpose() * (2 * mu) + Matrix2::Identity() * (lambda * j * (j - 1));
        const float ex = logf(fmaxf(svd.X, 1.e-6f)), ey = logf(fmaxf(svd.Y, 1.e-6f));
        const Matrix2 principal(2 * mu * ex + lambda * (ex + ey), 0, 0, 2 * mu * ey + lambda * (ex + ey));
        const Matrix2 sand = svd.U * principal * svd.U.Transpose();
        return snow * material.SnowFraction + sand * (1 - material.SnowFraction)
            + (c + c.Transpose()) * material.Viscosity;
    }

    R2_MPM_HD inline Matrix2 ProjectPlasticity(const Matrix2& trial, float& plasticVolume, const Material2D& material)
    {
        const Svd2 svd = Decompose(trial);
        float mu, lambda;
        Lame(material, plasticVolume, mu, lambda);
        const float sx = Limit(svd.X, 1 - material.CompressionYield, 1 + material.StretchYield);
        const float sy = Limit(svd.Y, 1 - material.CompressionYield, 1 + material.StretchYield);
        float ex = logf(fmaxf(svd.X, 1.e-6f)), ey = logf(fmaxf(svd.Y, 1.e-6f));
        const float trace = ex + ey;
        const float dx = ex - 0.5f * trace, dy = ey - 0.5f * trace;
        const float norm = sqrtf(dx * dx + dy * dy);
        const float sine = sinf(material.FrictionAngle * 0.01745329252f);
        const float alpha = 0.81649658093f * 2 * sine / (3 - sine);
        // Drucker-Prager return mapping, with dimension d = 2.
        const float plasticShear = norm + (lambda + mu) / mu * trace * alpha;
        if (trace >= 0) ex = ey = 0;
        else if (plasticShear > 0 && norm > 1.e-8f)
        {
            ex -= plasticShear * dx / norm;
            ey -= plasticShear * dy / norm;
        }
        const float x = expf(material.SnowFraction * logf(sx) + (1 - material.SnowFraction) * ex);
        const float y = expf(material.SnowFraction * logf(sy) + (1 - material.SnowFraction) * ey);
        plasticVolume = Limit(plasticVolume * fmaxf(trial.Determinant(), 1.e-6f) / fmaxf(x * y, 1.e-6f), 0.4f, 4.0f);
        return svd.U * Matrix2(x, 0, 0, y) * svd.V.Transpose();
    }

    R2_MPM_HD inline float Weight(int index, float distance)
    {
        const float x = distance - static_cast<float>(index);
        if (index == 0) return 0.5f * (1.5f - x) * (1.5f - x);
        if (index == 1) return 0.75f - x * x;
        return 0.5f * (1.5f + x) * (1.5f + x);
    }

    R2_MPM_HD inline float WeightGradient(int index, float distance)
    {
        if (index == 0) return distance - 1.5f;
        if (index == 1) return -2 * (distance - 1);
        return distance - 0.5f;
    }
}
