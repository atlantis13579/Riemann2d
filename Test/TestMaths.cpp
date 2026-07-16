#include "Test.h"

#include "Maths/Maths.h"
#include "Maths/Vector2.h"

using namespace Riemann2D;

R2_TEST("Vector2 arithmetic and products")
{
    const Vector2 a{2.0f, 3.0f};
    const Vector2 b{-4.0f, 5.0f};
    const Vector2 sum = a + b;
    R2_REQUIRE(NearlyEqual(sum.X, -2.0f));
    R2_REQUIRE(NearlyEqual(sum.Y, 8.0f));
    R2_REQUIRE(NearlyEqual(a.Dot(b), 7.0f));
    R2_REQUIRE(NearlyEqual(a.Cross(b), 22.0f));
}

R2_TEST("Vector2 normalization and rotation")
{
    const Vector2 normalized = Vector2{3.0f, 4.0f}.SafeUnit();
    R2_REQUIRE(NearlyEqual(normalized.Length(), 1.0f));
    R2_REQUIRE(NearlyEqual(Vector2{3.0f, 4.0f}.SquareLength(), 25.0f));
    R2_REQUIRE(Vector2::Zero().SafeUnit() == Vector2::Zero());

    Vector2 mutableVector{3.0f, 4.0f};
    R2_REQUIRE(NearlyEqual(mutableVector.SafeNormalize(), 5.0f));
    R2_REQUIRE(NearlyEqual(mutableVector.Length(), 1.0f));

    const Vector2 rotated = Vector2::UnitX().Rotated(0.5f * Pi);
    R2_REQUIRE(NearlyEqual(rotated.X, 0.0f, 1.0e-5f));
    R2_REQUIRE(NearlyEqual(rotated.Y, 1.0f, 1.0e-5f));
}

R2_TEST("Vector2 matches Riemann vector utility interface")
{
    Vector2 vector{2.0f, -3.0f};
    vector.SetZero();
    R2_REQUIRE(vector.IsZero());

    const Vector2 value{2.0f, -3.0f};
    R2_REQUIRE((value.Abs() == Vector2{2.0f, 3.0f}));
    R2_REQUIRE((value.Min(Vector2{1.0f, -4.0f}) == Vector2{1.0f, -4.0f}));
    R2_REQUIRE((value.Max(Vector2{1.0f, -4.0f}) == Vector2{2.0f, -3.0f}));
    R2_REQUIRE((Vector2::Lerp(Vector2::Zero(), Vector2::One(), 0.25f) == Vector2{0.25f, 0.25f}));
    R2_REQUIRE((Vector2{3.0f, 1.0f}.Project(Vector2::UnitX()) == Vector2{3.0f, 0.0f}));
    R2_REQUIRE(Vector2::UnitX().ParallelTo(Vector2{2.0f, 0.0f}));
}

R2_TEST("Scalar and vector cross products follow planar angular convention")
{
    const Vector2 velocity = Cross(2.0f, Vector2{3.0f, 0.0f});
    R2_REQUIRE(NearlyEqual(velocity.X, 0.0f));
    R2_REQUIRE(NearlyEqual(velocity.Y, 6.0f));
    const Vector2 reverse = Cross(Vector2{3.0f, 0.0f}, 2.0f);
    R2_REQUIRE(NearlyEqual(reverse.X, 0.0f));
    R2_REQUIRE(NearlyEqual(reverse.Y, -6.0f));
}
