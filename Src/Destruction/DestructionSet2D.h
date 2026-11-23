#pragma once

#include "Maths/Vector2.h"

#include <vector>

namespace Riemann2D
{
    class PhysicsWorld2D;
    class RigidBody2D;

    struct DestructionPiece2D
    {
        std::vector<Vector2> LocalVertices;
        Vector2 Position = Vector2::Zero();
        Vector2 Velocity = Vector2::Zero();
        float Angle = 0.0f;
        float AngularVelocity = 0.0f;
        RigidBody2D* Body = nullptr;
    };

    struct DestructionSettings2D
    {
        Vector2 Gravity{0.0f, -9.81f};
        float GroundHeight = -0.65f;
        float Restitution = 0.18f;
        float LinearDamping = 0.08f;
        float FragmentDensity = 1.0f;
        float FragmentSizeVariation = 0.0f;
    };

    class DestructionSet2D
    {
    public:
        void CreateBox(const Vector2& center, const Vector2& halfExtent, int pieceCount);
        void SetPhysicsWorld(PhysicsWorld2D* world) { World = world; }
        bool Fracture(const Vector2& impactPoint, float impulseStrength);
        void Step(float deltaTime);
        void Clear();

        bool IsFractured() const { return Fractured; }
        const std::vector<Vector2>& GetIntactVertices() const { return IntactVertices; }
        const std::vector<DestructionPiece2D>& GetPieces() const { return Pieces; }
        DestructionSettings2D& GetSettings() { return Settings; }

    private:
        void FractureRecursive(const std::vector<Vector2>& polygon, int pieceCount, int branch,
                               std::vector<std::vector<Vector2>>& output) const;
        void CreateFragmentBodies();

        DestructionSettings2D Settings;
        std::vector<Vector2> IntactVertices;
        std::vector<DestructionPiece2D> Pieces;
        Vector2 BoxCenter = Vector2::Zero();
        PhysicsWorld2D* World = nullptr;
        int TargetPieceCount = 1;
        bool Fractured = false;
    };
}
