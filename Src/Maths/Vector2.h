#pragma once

#include "Maths.h"

#include <cstdlib>
#include <limits>

namespace Riemann2D
{
    struct Vector2
    {
        float X = 0.0f;
        float Y = 0.0f;

        constexpr Vector2() = default;
        explicit constexpr Vector2(float value) : X(value), Y(value) {}
        constexpr Vector2(float x, float y) : X(x), Y(y) {}

        void SetZero() { X = Y = 0.0f; }

        constexpr Vector2 operator+() const { return *this; }
        constexpr Vector2 operator-() const { return {-X, -Y}; }
        constexpr Vector2 operator+(const Vector2& rhs) const { return {X + rhs.X, Y + rhs.Y}; }
        constexpr Vector2 operator-(const Vector2& rhs) const { return {X - rhs.X, Y - rhs.Y}; }
        constexpr Vector2 operator*(const Vector2& rhs) const { return {X * rhs.X, Y * rhs.Y}; }
        constexpr Vector2 operator/(const Vector2& rhs) const { return {X / rhs.X, Y / rhs.Y}; }
        constexpr Vector2 operator+(float value) const { return {X + value, Y + value}; }
        constexpr Vector2 operator-(float value) const { return {X - value, Y - value}; }
        constexpr Vector2 operator*(float scale) const { return {X * scale, Y * scale}; }
        constexpr Vector2 operator/(float scale) const { return {X / scale, Y / scale}; }

        Vector2& operator+=(const Vector2& rhs) { X += rhs.X; Y += rhs.Y; return *this; }
        Vector2& operator-=(const Vector2& rhs) { X -= rhs.X; Y -= rhs.Y; return *this; }
        Vector2& operator*=(const Vector2& rhs) { X *= rhs.X; Y *= rhs.Y; return *this; }
        Vector2& operator/=(const Vector2& rhs) { X /= rhs.X; Y /= rhs.Y; return *this; }
        Vector2& operator+=(float value) { X += value; Y += value; return *this; }
        Vector2& operator-=(float value) { X -= value; Y -= value; return *this; }
        Vector2& operator*=(float scale) { X *= scale; Y *= scale; return *this; }
        Vector2& operator/=(float scale) { X /= scale; Y /= scale; return *this; }

        constexpr bool operator>(float value) const { return X > value && Y > value; }
        constexpr bool operator>=(float value) const { return X >= value && Y >= value; }
        constexpr bool operator<(float value) const { return X < value && Y < value; }
        constexpr bool operator<=(float value) const { return X <= value && Y <= value; }
        constexpr bool operator>(const Vector2& rhs) const { return X > rhs.X && Y > rhs.Y; }
        constexpr bool operator>=(const Vector2& rhs) const { return X >= rhs.X && Y >= rhs.Y; }
        constexpr bool operator<(const Vector2& rhs) const { return X < rhs.X && Y < rhs.Y; }
        constexpr bool operator<=(const Vector2& rhs) const { return X <= rhs.X && Y <= rhs.Y; }
        constexpr bool operator==(const Vector2& rhs) const { return X == rhs.X && Y == rhs.Y; }
        constexpr bool operator!=(const Vector2& rhs) const { return X != rhs.X || Y != rhs.Y; }

        constexpr float Dot(const Vector2& rhs) const { return X * rhs.X + Y * rhs.Y; }
        constexpr float Cross(const Vector2& rhs) const { return X * rhs.Y - Y * rhs.X; }
        constexpr float SquareLength() const { return Dot(*this); }
        float Length() const { return std::sqrt(SquareLength()); }

        Vector2 Unit() const
        {
            return *this / Length();
        }

        Vector2 SafeUnit() const
        {
            const float length = Length();
            return length > std::numeric_limits<float>::epsilon() ? *this / length : Zero();
        }

        float Normalize()
        {
            const float length = Length();
            *this /= length;
            return length;
        }

        float SafeNormalize()
        {
            const float length = Length();
            if (length <= std::numeric_limits<float>::epsilon())
            {
                SetZero();
                return 0.0f;
            }
            *this /= length;
            return length;
        }

        Vector2 Rotated(float radians) const
        {
            const float cosine = std::cos(radians);
            const float sine = std::sin(radians);
            return {cosine * X - sine * Y, sine * X + cosine * Y};
        }

        constexpr Vector2 PerpendicularCCW() const { return {-Y, X}; }

        operator const float*() const { return Data(); }
        operator float*() { return Data(); }
        const float* Data() const { return &X; }
        float* Data() { return &X; }
        const float& operator[](int index) const { return Data()[index]; }
        float& operator[](int index) { return Data()[index]; }

        bool IsZero() const { return SquareLength() <= std::numeric_limits<float>::epsilon(); }
        bool ParallelTo(const Vector2& rhs) const
        {
            return Cross(rhs) * Cross(rhs) < std::numeric_limits<float>::epsilon();
        }

        int LargestAxis() const { return Y > X ? 1 : 0; }

        Vector2 Project(const Vector2& to) const
        {
            if (to.SquareLength() < std::numeric_limits<float>::epsilon()) return Zero();
            return to * (Dot(to) / to.Dot(to));
        }

        Vector2 Abs() const { return {std::abs(X), std::abs(Y)}; }
        Vector2 Min(const Vector2& rhs) const { return {std::min(X, rhs.X), std::min(Y, rhs.Y)}; }
        Vector2 Max(const Vector2& rhs) const { return {std::max(X, rhs.X), std::max(Y, rhs.Y)}; }

        static Vector2 Lerp(const Vector2& start, const Vector2& end, float t)
        {
            return start * (1.0f - t) + end * t;
        }

        static Vector2 UnitLerp(const Vector2& start, const Vector2& end, float t)
        {
            return Lerp(start, end, t).Unit();
        }

        static Vector2 Slerp(const Vector2& start, const Vector2& end, float t)
        {
            const float innerProduct = Clamp(start.Dot(end), -1.0f, 1.0f);
            const float angle = std::acos(innerProduct) * t;
            const Vector2 base = (end - start * innerProduct).Unit();
            return start * std::cos(angle) + base * std::sin(angle);
        }

        static constexpr Vector2 InfMin()
        {
            return {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
        }

        static constexpr Vector2 InfMax()
        {
            return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        }
        static constexpr Vector2 Zero() { return {}; }
        static constexpr Vector2 One() { return {1.0f, 1.0f}; }
        static constexpr Vector2 UnitX() { return {1.0f, 0.0f}; }
        static constexpr Vector2 UnitY() { return {0.0f, 1.0f}; }
        static Vector2 Random()
        {
            return {static_cast<float>(std::rand()) / RAND_MAX,
                    static_cast<float>(std::rand()) / RAND_MAX};
        }
    };

    constexpr Vector2 operator*(float scale, const Vector2& vector)
    {
        return vector * scale;
    }

    constexpr Vector2 Cross(const Vector2& vector, float scalar)
    {
        return -scalar * vector.PerpendicularCCW();
    }

    constexpr Vector2 Cross(float scalar, const Vector2& vector)
    {
        return scalar * vector.PerpendicularCCW();
    }
}
