#include "Ragdoll2D.h"

#include "Collision/Box2.h"
#include "Collision/Capsule2.h"
#include "Collision/Sphere2.h"
#include "Dynamics/Constraints2D.h"

#include <memory>

namespace Riemann2D
{
    bool Ragdoll2D::Build(PhysicsWorld2D& world, const Vector2& origin, float scale)
    {
        Clear();
        if (scale <= 0.0f) return false;

        auto createPart = [&](const char* name, std::shared_ptr<const Shape2> shape,
                              const Vector2& offset, float angle, float density) -> RigidBody2D*
        {
            RigidBodyDesc2D description;
            description.Shape = std::move(shape);
            description.Transform.Position = origin + scale * offset;
            description.Transform.Angle = angle;
            description.Density = density;
            description.Friction = 0.65f;
            description.LinearDamping = 0.08f;
            description.AngularDamping = 0.12f;
            RigidBody2D* body = &world.CreateBody(description);
            Parts.push_back({name, body});
            return body;
        };

        RigidBody2D* torso = createPart("Torso",
            std::make_shared<Box2>(scale * Vector2{0.48f, 0.95f}), {0.0f, 0.0f}, 0.0f, 1.4f);
        RigidBody2D* head = createPart("Head",
            std::make_shared<Sphere2>(0.43f * scale), {0.0f, 1.38f}, 0.0f, 0.9f);
        RigidBody2D* leftUpperArm = createPart("Left upper arm",
            std::make_shared<Capsule2>(0.45f * scale, 0.16f * scale), {-0.93f, 0.65f}, 0.0f, 0.8f);
        RigidBody2D* leftLowerArm = createPart("Left lower arm",
            std::make_shared<Capsule2>(0.42f * scale, 0.14f * scale), {-1.80f, 0.65f}, 0.0f, 0.7f);
        RigidBody2D* rightUpperArm = createPart("Right upper arm",
            std::make_shared<Capsule2>(0.45f * scale, 0.16f * scale), {0.93f, 0.65f}, 0.0f, 0.8f);
        RigidBody2D* rightLowerArm = createPart("Right lower arm",
            std::make_shared<Capsule2>(0.42f * scale, 0.14f * scale), {1.80f, 0.65f}, 0.0f, 0.7f);
        RigidBody2D* leftUpperLeg = createPart("Left upper leg",
            std::make_shared<Capsule2>(0.58f * scale, 0.19f * scale), {-0.27f, -1.50f}, 0.5f * Pi, 1.2f);
        RigidBody2D* leftLowerLeg = createPart("Left lower leg",
            std::make_shared<Capsule2>(0.54f * scale, 0.16f * scale), {-0.27f, -2.62f}, 0.5f * Pi, 1.0f);
        RigidBody2D* rightUpperLeg = createPart("Right upper leg",
            std::make_shared<Capsule2>(0.58f * scale, 0.19f * scale), {0.27f, -1.50f}, 0.5f * Pi, 1.2f);
        RigidBody2D* rightLowerLeg = createPart("Right lower leg",
            std::make_shared<Capsule2>(0.54f * scale, 0.16f * scale), {0.27f, -2.62f}, 0.5f * Pi, 1.0f);

        auto addJoint = [&](RigidBody2D* bodyA, RigidBody2D* bodyB,
                            const Vector2& anchorA, const Vector2& anchorB,
                            float lower, float upper)
        {
            RevoluteConstraintDesc2D description;
            description.BodyA = bodyA;
            description.BodyB = bodyB;
            description.LocalAnchorA = scale * anchorA;
            description.LocalAnchorB = scale * anchorB;
            description.EnableLimit = true;
            description.LowerAngle = lower;
            description.UpperAngle = upper;
            description.Settings.MaxPositionCorrection = 0.12f * scale;
            world.AddConstraint(std::make_unique<RevoluteConstraint2D>(description));
        };

        addJoint(torso, head, {0.0f, 0.95f}, {0.0f, -0.43f}, -0.45f, 0.45f);
        addJoint(torso, leftUpperArm, {-0.48f, 0.65f}, {0.45f, 0.0f}, -1.4f, 1.0f);
        addJoint(leftUpperArm, leftLowerArm, {-0.45f, 0.0f}, {0.42f, 0.0f}, -1.8f, 0.15f);
        addJoint(torso, rightUpperArm, {0.48f, 0.65f}, {-0.45f, 0.0f}, -1.0f, 1.4f);
        addJoint(rightUpperArm, rightLowerArm, {0.45f, 0.0f}, {-0.42f, 0.0f}, -0.15f, 1.8f);
        addJoint(torso, leftUpperLeg, {-0.27f, -0.95f}, {0.58f, 0.0f}, -0.8f, 0.8f);
        addJoint(leftUpperLeg, leftLowerLeg, {-0.58f, 0.0f}, {0.54f, 0.0f}, -0.1f, 1.5f);
        addJoint(torso, rightUpperLeg, {0.27f, -0.95f}, {0.58f, 0.0f}, -0.8f, 0.8f);
        addJoint(rightUpperLeg, rightLowerLeg, {-0.58f, 0.0f}, {0.54f, 0.0f}, -0.1f, 1.5f);
        return true;
    }
}
