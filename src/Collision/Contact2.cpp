#include "Contact2.h"

#include "Collision/Box2.h"
#include "Collision/GJK2.h"
#include "Collision/Convex2.h"
#include "Collision/Sphere2.h"
#include "Collision/TriangleMesh2.h"
#include "Dynamics/RigidBody2D.h"
#include "Maths/Maths.h"

#include <array>
#include <cfloat>

namespace Riemann2D
{
    namespace
    {
        Vector2 WorldSupport(const RigidBody2D& body, const Vector2& direction)
        {
            const Transform2& transform = body.GetTransform();
            return transform.TransformPoint(body.GetShape().Support(transform.InverseTransformVector(direction)));
        }

        Vector2 WorldBoxContactSupport(const RigidBody2D& body, const Vector2& direction)
        {
            const auto& box = static_cast<const Box2&>(body.GetShape());
            const Transform2& transform = body.GetTransform();
            const Vector2 localDirection = transform.InverseTransformVector(direction);
            const Vector2 halfExtent = box.GetHalfExtent();
            const auto supportCoordinate = [](float axisDirection, float halfExtentValue)
            {
                if (std::abs(axisDirection) <= Epsilon) return 0.0f;
                return axisDirection > 0.0f ? halfExtentValue : -halfExtentValue;
            };
            return transform.TransformPoint({supportCoordinate(localDirection.X, halfExtent.X),
                                             supportCoordinate(localDirection.Y, halfExtent.Y)});
        }

        bool SphereSphere(RigidBody2D& bodyA, RigidBody2D& bodyB, Contact2& contact)
        {
            const auto& sphereA = static_cast<const Sphere2&>(bodyA.GetShape());
            const auto& sphereB = static_cast<const Sphere2&>(bodyB.GetShape());
            const Vector2 delta = bodyB.GetTransform().Position - bodyA.GetTransform().Position;
            const float distanceSquared = delta.SquareLength();
            const float radius = sphereA.GetRadius() + sphereB.GetRadius();
            if (distanceSquared > radius * radius)
            {
                return false;
            }
            const float distance = std::sqrt(distanceSquared);
            contact.Normal = distance > Epsilon ? delta / distance : Vector2::UnitX();
            contact.Penetration = radius - distance;
            contact.Point = bodyA.GetTransform().Position + contact.Normal *
                            (sphereA.GetRadius() - 0.5f * contact.Penetration);
            return true;
        }

        bool BoxSphere(RigidBody2D& boxBody, RigidBody2D& sphereBody, Contact2& contact)
        {
            const auto& box = static_cast<const Box2&>(boxBody.GetShape());
            const auto& sphere = static_cast<const Sphere2&>(sphereBody.GetShape());
            const Transform2& boxTransform = boxBody.GetTransform();
            const Vector2 localCenter = boxTransform.InverseTransformPoint(sphereBody.GetTransform().Position);
            const Vector2 half = box.GetHalfExtent();
            Vector2 closest{Clamp(localCenter.X, -half.X, half.X), Clamp(localCenter.Y, -half.Y, half.Y)};
            Vector2 localDelta = localCenter - closest;
            float distanceSquared = localDelta.SquareLength();

            if (distanceSquared <= Epsilon)
            {
                const float distanceX = half.X - std::abs(localCenter.X);
                const float distanceY = half.Y - std::abs(localCenter.Y);
                if (distanceX < distanceY)
                {
                    localDelta = {localCenter.X >= 0.0f ? 1.0f : -1.0f, 0.0f};
                    closest.X = localDelta.X * half.X;
                    contact.Penetration = sphere.GetRadius() + distanceX;
                }
                else
                {
                    localDelta = {0.0f, localCenter.Y >= 0.0f ? 1.0f : -1.0f};
                    closest.Y = localDelta.Y * half.Y;
                    contact.Penetration = sphere.GetRadius() + distanceY;
                }
            }
            else
            {
                if (distanceSquared > sphere.GetRadius() * sphere.GetRadius())
                {
                    return false;
                }
                const float distance = std::sqrt(distanceSquared);
                localDelta = localDelta / distance;
                contact.Penetration = sphere.GetRadius() - distance;
            }
            contact.Normal = boxTransform.TransformVector(localDelta);
            contact.Point = boxTransform.TransformPoint(closest);
            return true;
        }

        bool BoxBox(RigidBody2D& bodyA, RigidBody2D& bodyB, Contact2& contact)
        {
            const Transform2& transformA = bodyA.GetTransform();
            const Transform2& transformB = bodyB.GetTransform();
            const std::array<Vector2, 4> axes = {
                transformA.TransformVector(Vector2::UnitX()), transformA.TransformVector(Vector2::UnitY()),
                transformB.TransformVector(Vector2::UnitX()), transformB.TransformVector(Vector2::UnitY())
            };
            float minimumOverlap = FLT_MAX;
            Vector2 bestAxis = Vector2::UnitX();
            std::size_t bestAxisIndex = 0;
            for (std::size_t axisIndex = 0; axisIndex < axes.size(); ++axisIndex)
            {
                Vector2 axis = axes[axisIndex];
                axis = axis.SafeUnit();
                const float maximumA = WorldSupport(bodyA, axis).Dot(axis);
                const float minimumA = WorldSupport(bodyA, -axis).Dot(axis);
                const float maximumB = WorldSupport(bodyB, axis).Dot(axis);
                const float minimumB = WorldSupport(bodyB, -axis).Dot(axis);
                const float overlap = std::min(maximumA, maximumB) - std::max(minimumA, minimumB);
                if (overlap < 0.0f)
                {
                    return false;
                }
                if (overlap < minimumOverlap)
                {
                    minimumOverlap = overlap;
                    bestAxis = axis;
                    bestAxisIndex = axisIndex;
                }
            }
            if ((transformB.Position - transformA.Position).Dot(bestAxis) < 0.0f)
            {
                bestAxis = -bestAxis;
            }
            contact.Normal = bestAxis;
            contact.Penetration = minimumOverlap;
            const Vector2 supportA = WorldBoxContactSupport(bodyA, bestAxis);
            const Vector2 supportB = WorldBoxContactSupport(bodyB, -bestAxis);
            // Preserve the incident vertex's tangential coordinate. Averaging the body
            // centers places every symmetric box contact on the center line and erases
            // the torque generated when a tilted box first touches a flat surface.
            contact.Point = bestAxisIndex < 2
                ? supportB + 0.5f * minimumOverlap * bestAxis
                : supportA - 0.5f * minimumOverlap * bestAxis;
            return true;
        }

        bool ConvexPair(const Shape2& shapeA, const Transform2& transformA,
                        const Shape2& shapeB, const Transform2& transformB,
                        Contact2& contact)
        {
            ConvexCollisionResult2 result;
            if (!CollideConvex2(shapeA, transformA, shapeB, transformB, result))
            {
                return false;
            }
            contact.Normal = result.Normal.SafeUnit();
            contact.Point = result.Point;
            contact.Penetration = result.Penetration;
            return true;
        }

        Convex2 MakeTriangle(const TriangleMesh2& mesh, std::size_t triangleIndex)
        {
            const TriangleMesh2::Triangle triangle = mesh.GetTriangle(triangleIndex);
            return Convex2({triangle[0], triangle[1], triangle[2]});
        }

        bool PrimitiveMesh(RigidBody2D& primitiveBody, RigidBody2D& meshBody,
                           bool primitiveIsA, Contact2& contact)
        {
            const auto& mesh = static_cast<const TriangleMesh2&>(meshBody.GetShape());
            bool found = false;
            Contact2 best = contact;
            for (std::size_t triangleIndex = 0; triangleIndex < mesh.GetTriangleCount(); ++triangleIndex)
            {
                const Convex2 triangle = MakeTriangle(mesh, triangleIndex);
                Contact2 candidate = contact;
                const bool collided = primitiveIsA
                    ? ConvexPair(primitiveBody.GetShape(), primitiveBody.GetTransform(),
                                 triangle, meshBody.GetTransform(), candidate)
                    : ConvexPair(triangle, meshBody.GetTransform(), primitiveBody.GetShape(),
                                 primitiveBody.GetTransform(), candidate);
                if (collided && (!found || candidate.Penetration > best.Penetration))
                {
                    found = true;
                    best = candidate;
                    if (primitiveIsA)
                    {
                        best.TriangleIndexB = static_cast<std::int32_t>(triangleIndex);
                    }
                    else
                    {
                        best.TriangleIndexA = static_cast<std::int32_t>(triangleIndex);
                    }
                }
            }
            if (found)
            {
                contact = best;
            }
            return found;
        }

        bool MeshMesh(RigidBody2D& bodyA, RigidBody2D& bodyB, Contact2& contact)
        {
            const auto& meshA = static_cast<const TriangleMesh2&>(bodyA.GetShape());
            const auto& meshB = static_cast<const TriangleMesh2&>(bodyB.GetShape());
            bool found = false;
            Contact2 best = contact;
            for (std::size_t indexA = 0; indexA < meshA.GetTriangleCount(); ++indexA)
            {
                const Convex2 triangleA = MakeTriangle(meshA, indexA);
                for (std::size_t indexB = 0; indexB < meshB.GetTriangleCount(); ++indexB)
                {
                    const Convex2 triangleB = MakeTriangle(meshB, indexB);
                    Contact2 candidate = contact;
                    if (ConvexPair(triangleA, bodyA.GetTransform(), triangleB, bodyB.GetTransform(), candidate) &&
                        (!found || candidate.Penetration > best.Penetration))
                    {
                        found = true;
                        best = candidate;
                        best.TriangleIndexA = static_cast<std::int32_t>(indexA);
                        best.TriangleIndexB = static_cast<std::int32_t>(indexB);
                    }
                }
            }
            if (found)
            {
                contact = best;
            }
            return found;
        }
    }

    bool GenerateContact(RigidBody2D& bodyA, RigidBody2D& bodyB, Contact2& contact)
    {
        contact = {};
        contact.BodyA = &bodyA;
        contact.BodyB = &bodyB;
        const ShapeType2 typeA = bodyA.GetShape().GetType();
        const ShapeType2 typeB = bodyB.GetShape().GetType();
        if (typeA == ShapeType2::TriangleMesh2 && typeB == ShapeType2::TriangleMesh2)
        {
            return MeshMesh(bodyA, bodyB, contact);
        }
        if (typeB == ShapeType2::TriangleMesh2)
        {
            return PrimitiveMesh(bodyA, bodyB, true, contact);
        }
        if (typeA == ShapeType2::TriangleMesh2)
        {
            return PrimitiveMesh(bodyB, bodyA, false, contact);
        }
        if (typeA == ShapeType2::Sphere2 && typeB == ShapeType2::Sphere2)
        {
            return SphereSphere(bodyA, bodyB, contact);
        }
        if (typeA == ShapeType2::Box && typeB == ShapeType2::Box)
        {
            return BoxBox(bodyA, bodyB, contact);
        }
        if (typeA == ShapeType2::Box && typeB == ShapeType2::Sphere2)
        {
            return BoxSphere(bodyA, bodyB, contact);
        }
        if (typeA == ShapeType2::Sphere2 && typeB == ShapeType2::Box)
        {
            const bool result = BoxSphere(bodyB, bodyA, contact);
            contact.BodyA = &bodyA;
            contact.BodyB = &bodyB;
            contact.Normal = -contact.Normal;
            return result;
        }
        return ConvexPair(bodyA.GetShape(), bodyA.GetTransform(), bodyB.GetShape(), bodyB.GetTransform(), contact);
    }
}
