#pragma once

#include "Render/ImmediateUI.h"
#include "Render/Renderer2.h"
#include "Dynamics/PhysicsWorld2D.h"
#include "SoftBody/RopeSimulation2D.h"
#include "Fluid/FluidSimulation2D.h"
#include "Fluid/AerodynamicsSimulation2D.h"
#include "Destruction/DestructionSet2D.h"
#include "Dynamics/Ragdoll2D.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Riemann2D::Viewer
{
    class Viewer
    {
    public:
        bool Initialize(void* nativeWindow, int width, int height);
        void Resize(int width, int height);
        void Tick(float deltaTime, const Render::InterfaceInput& input);
        void Shutdown();

    private:
        enum SceneIndex
        {
            ShapeGalleryScene,
            JointChainScene,
            RopeScene,
            FluidScene,
            AirScene,
            DestructionScene,
            RagdollScene
        };

        struct Scene
        {
            std::string Name;
        };

        struct FluidTankWall
        {
            RigidBody2D* Body = nullptr;
            Transform2 LocalTransform;
        };

        std::shared_ptr<Render::FramePacket> BuildFrame(const Render::InterfaceInput& input);
        Render::DrawPacket BuildWorldDraw() const;
        Render::DrawPacket BuildGridDraw() const;
        void RebuildPhysicsScene();
        void HandleFluidInteraction(const Render::InterfaceInput& input);
        void CancelFluidInteraction();
        Vector2 ScreenToWorld(const Render::InterfaceInput& input) const;
        RigidBody2D* PickFluidBody(const Vector2& worldPosition) const;
        void UpdateFluidTankTransform();
        void ConfineFluidBodiesToTank();

        Render::Renderer2 Renderer;
        Render::ImmediateUI Interface;
        std::vector<Scene> Scenes;
        std::uint64_t NextFrameId = 1;
        std::uint64_t LastSubmittedFrame = 0;
        int Width = 1280;
        int Height = 720;
        int SelectedScene = 0;
        int SelectedSolver = 0;
        int SelectedBackend = 0;
        bool RenderFluidSurface = true;
        float Gravity = -9.81f;
        float AnimationTime = 0.0f;
        PhysicsWorld2D World;
        std::unique_ptr<RopeSimulation2D> Rope;
        std::unique_ptr<FluidSimulation2D> Fluid;
        std::vector<FluidTankWall> FluidTankWalls;
        std::vector<RigidBody2D*> FluidBodies;
        Transform2 FluidTankTransform;
        MouseConstraint2D* ActiveMouseConstraint = nullptr;
        bool RotatingFluidTank = false;
        float FluidTankStartAngle = 0.0f;
        float MouseStartAngle = 0.0f;
        std::unique_ptr<AerodynamicsSimulation2D> Air;
        std::unique_ptr<DestructionSet2D> Destruction;
        RigidBody2D* DestructionProjectile = nullptr;
        std::unique_ptr<Ragdoll2D> Ragdoll;
    };
}
