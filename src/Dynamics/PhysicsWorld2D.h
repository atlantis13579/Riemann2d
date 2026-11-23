#pragma once

#include "Collision/Contact2.h"
#include "Collision/SAT2.h"
#include "Collision/NarrowPhase2.h"
#include "Dynamics/RobotController2D.h"
#include "Dynamics/Constraints2D.h"
#include "Dynamics/RigidBody2D.h"
#include "Dynamics/ContactSolver2D.h"
#include "Dynamics/ForceField2D.h"

#include <memory>
#include <vector>

namespace Riemann2D
{
    struct PhysicsWorldSettings2D
    {
        Vector2 Gravity{0.0f, -9.81f};
        int VelocityIterations = 8;
        int PositionIterations = 3;
        ContactSolverType2D SolverType = ContactSolverType2D::SequentialImpulse;
        SolverBackend2D SolverBackend = SolverBackend2D::ScalarCPU;
    };

    class PhysicsWorld2D
    {
    public:
        explicit PhysicsWorld2D(const PhysicsWorldSettings2D& settings = {});

        RigidBody2D& CreateBody(const RigidBodyDesc2D& description);
        RobotController2D& CreateRobotController(const RobotControllerSettings2D& settings, const Vector2& position);
        IConstraint2D& AddConstraint(std::unique_ptr<IConstraint2D> constraint);
        bool RemoveConstraint(IConstraint2D* constraint);
        ForceField2D& AddForceField(std::unique_ptr<ForceField2D> field);
        void Clear();
        void Step(float deltaTime);
        void SetGravity(const Vector2& gravity) { Settings.Gravity = gravity; }
        void SetContactSolver(ContactSolverType2D type, SolverBackend2D backend)
        {
            Settings.SolverType = type;
            Settings.SolverBackend = backend;
        }
        float GetLastSolverResidual() const { return ContactSolver.GetLastResidual(); }

        const std::vector<std::unique_ptr<RigidBody2D>>& GetBodies() const { return Bodies; }
        const std::vector<Contact2>& GetContacts() const { return Contacts; }
        std::size_t GetConstraintCount() const { return Constraints.size(); }
        std::size_t GetForceFieldCount() const { return ForceFields.size(); }

    private:
        void FindContacts();
        void SolvePosition(Contact2& contact);

        PhysicsWorldSettings2D Settings;
        std::vector<std::unique_ptr<RigidBody2D>> Bodies;
        std::vector<std::unique_ptr<RobotController2D>> Robots;
        std::vector<std::unique_ptr<IConstraint2D>> Constraints;
        std::vector<std::unique_ptr<ForceField2D>> ForceFields;
        std::vector<Contact2> Contacts;
        SweepAndPruneBroadPhase2 BroadPhase;
        NarrowPhase2 NarrowPhase;
        std::vector<BroadPhaseProxy2> Proxies;
        std::vector<BroadPhasePair2> CandidatePairs;
        ContactSolver2D ContactSolver;
    };
}
