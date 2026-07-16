#pragma once

#include "Vector2.h"

namespace Riemann2D
{
    struct Transform2
    {
        Vector2 Position = Vector2::Zero();
        float Angle = 0.0f;

        Vector2 TransformVector(const Vector2& localVector) const
        {
            return localVector.Rotated(Angle);
        }

        Vector2 TransformPoint(const Vector2& localPoint) const
        {
            return Position + TransformVector(localPoint);
        }

        Vector2 InverseTransformVector(const Vector2& worldVector) const
        {
            return worldVector.Rotated(-Angle);
        }

        Vector2 InverseTransformPoint(const Vector2& worldPoint) const
        {
            return InverseTransformVector(worldPoint - Position);
        }
    };
}
