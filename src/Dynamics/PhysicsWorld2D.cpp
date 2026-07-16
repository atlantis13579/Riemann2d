#include "PhysicsWorld2D.h"

#include "Maths/Maths.h"

#include <algorithm>
#include <cmath>

namespace Riemann2D
{
    PhysicsWorld2D::PhysicsWorld2D(const PhysicsWorldSettings2D& settings)
        : Settings(settings)
    {
    }

    RigidBody2D& PhysicsWorld2D::CreateBody(const RigidBodyDesc2D& description)
    {
        Bodies.push_back(std::make_unique<RigidBody2D>(description));
        return *Bodies.back();
    }

    IConstraint2D& PhysicsWorld2D::AddConstraint(std::unique_ptr<IConstraint2D> constraint)
    {
        Constraints.push_back(std::move(constraint));
        return *Constraints.back();
    }

    bool PhysicsWorld2D::RemoveConstraint(IConstraint2D* constraint)
    {
        const auto found = std::find_if(Constraints.begin(), Constraints.end(),
            [constraint](const std::unique_ptr<IConstraint2D>& candidate)
            {
                return candidate.get() == constraint;
            });
        if (found == Constraints.end()) return false;
        Constraints.erase(found);
        return true;
    }

    ForceField2D& PhysicsWorld2D::AddForceField(std::unique_ptr<ForceField2D> field)
    {
        ForceFields.push_back(std::move(field));
        return *ForceFields.back();
    }

    void PhysicsWorld2D::Clear()
    {
        Contacts.clear();
        Constraints.clear();
        ForceFields.clear();
        Bodies.clear();
    }

    void PhysicsWorld2D::Step(float deltaTime)
    {
        if (deltaTime <= 0.0f)
        {
            return;
        }
        deltaTime = std::min(deltaTime, 1.0f / 20.0f);
        for (const std::unique_ptr<ForceField2D>& field : ForceFields)
        {
            field->Update(deltaTime);
            for (const std::unique_ptr<RigidBody2D>& body : Bodies)
            {
                if (body->Type == BodyType2D::Dynamic) field->ApplyForce(*body);
            }
        }
        for (const std::unique_ptr<RigidBody2D>& body : Bodies)
        {
            if (body->Type == BodyType2D::Dynamic)
            {
                body->LinearVelocity += deltaTime *
                    (Settings.Gravity * body->GravityScale + body->Force * body->InverseMass);
                body->AngularVelocity += deltaTime * body->Torque * body->InverseInertia;
                body->LinearVelocity *= 1.0f / (1.0f + deltaTime * body->LinearDamping);
                body->AngularVelocity *= 1.0f / (1.0f + deltaTime * body->AngularDamping);
            }
            body->Force = Vector2::Zero();
            body->Torque = 0.0f;
        }

        FindContacts();
        ContactSolver.Solve(Contacts, Settings.SolverType, Settings.SolverBackend,
                            Settings.VelocityIterations);
        for (int iteration = 0; iteration < Settings.VelocityIterations; ++iteration)
        {
            for (const std::unique_ptr<IConstraint2D>& constraint : Constraints)
            {
                constraint->SolveVelocity(deltaTime);
            }
        }

        for (const std::unique_ptr<RigidBody2D>& body : Bodies)
        {
            if (body->Type != BodyType2D::Static)
            {
                body->Pose.Position += deltaTime * body->LinearVelocity;
                body->Pose.Angle += deltaTime * body->AngularVelocity;
            }
        }

        for (int iteration = 0; iteration < Settings.PositionIterations; ++iteration)
        {
            FindContacts();
            for (Contact2& contact : Contacts)
            {
                SolvePosition(contact);
            }
            for (const std::unique_ptr<IConstraint2D>& constraint : Constraints)
            {
                constraint->SolvePosition();
            }
        }
    }

    void PhysicsWorld2D::FindContacts()
    {
        Proxies.clear();
        Proxies.reserve(Bodies.size());
        for (std::size_t index = 0; index < Bodies.size(); ++index)
        {
            const RigidBody2D& body = *Bodies[index];
            Proxies.push_back({static_cast<std::uint32_t>(index),
                               body.Shape->ComputeAABB(body.Pose),
                               body.Type != BodyType2D::Dynamic});
        }
        BroadPhase.ComputePairs(Proxies, CandidatePairs);
        NarrowPhase.GenerateContacts(Bodies, CandidatePairs, Contacts);
    }

    void PhysicsWorld2D::SolvePosition(Contact2& contact)
    {
        constexpr float slop = 0.005f;
        constexpr float correctionRatio = 0.35f;
        RigidBody2D& bodyA = *contact.BodyA;
        RigidBody2D& bodyB = *contact.BodyB;
        const float inverseMass = bodyA.InverseMass + bodyB.InverseMass;
        if (inverseMass <= Epsilon)
        {
            return;
        }
        const float magnitude = correctionRatio * std::max(contact.Penetration - slop, 0.0f) / inverseMass;
        const Vector2 correction = magnitude * contact.Normal;
        if (bodyA.Type == BodyType2D::Dynamic) bodyA.Pose.Position -= correction * bodyA.InverseMass;
        if (bodyB.Type == BodyType2D::Dynamic) bodyB.Pose.Position += correction * bodyB.InverseMass;
    }
}
