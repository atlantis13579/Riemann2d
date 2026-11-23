#include "DestructionSet2D.h"

#include "DestructionCut2D.h"

#include "Collision/Convex2.h"
#include "Dynamics/PhysicsWorld2D.h"
#include "Maths/Maths.h"

#include <algorithm>
#include <cmath>

namespace Riemann2D
{
    void DestructionSet2D::CreateBox(const Vector2& center, const Vector2& halfExtent,
                                        int pieceCount)
    {
        Clear();
        BoxCenter = center;
        TargetPieceCount = std::max(1, std::min(pieceCount, 32));
        IntactVertices = {
            center + Vector2{-halfExtent.X, -halfExtent.Y},
            center + Vector2{halfExtent.X, -halfExtent.Y},
            center + Vector2{halfExtent.X, halfExtent.Y},
            center + Vector2{-halfExtent.X, halfExtent.Y}
        };
    }

    void DestructionSet2D::FractureRecursive(const std::vector<Vector2>& polygon, int pieceCount,
                                                int branch,
                                                std::vector<std::vector<Vector2>>& output) const
    {
        if (pieceCount <= 1)
        {
            output.push_back(polygon);
            return;
        }
        const int negativePieceCount = pieceCount / 2;
        const int positivePieceCount = pieceCount - negativePieceCount;
        std::vector<Vector2> negative;
        std::vector<Vector2> positive;
        if (!FindCompactDestructionSplit2D(polygon, negativePieceCount, positivePieceCount,
                                           branch, negative, positive, Settings.FragmentSizeVariation))
        {
            output.push_back(polygon);
            return;
        }
        FractureRecursive(negative, negativePieceCount, branch * 2 + 1, output);
        FractureRecursive(positive, positivePieceCount, branch * 2 + 2, output);
    }

    bool DestructionSet2D::Fracture(const Vector2& impactPoint, float impulseStrength)
    {
        if (Fractured || IntactVertices.size() < 3) return false;
        std::vector<std::vector<Vector2>> fragments;
        FractureRecursive(IntactVertices, TargetPieceCount, 1, fragments);
        for (const std::vector<Vector2>& vertices : fragments)
        {
            DestructionPiece2D piece;
            piece.Position = PolygonCentroid2D(vertices);
            piece.LocalVertices.reserve(vertices.size());
            for (const Vector2& vertex : vertices) piece.LocalVertices.push_back(vertex - piece.Position);
            Vector2 direction = (piece.Position - impactPoint).SafeUnit();
            if (direction.SquareLength() <= Epsilon) direction = Vector2::UnitX();
            piece.Velocity = impulseStrength * direction;
            piece.AngularVelocity = 0.35f * impulseStrength *
                std::sin(piece.Position.X * 1.73f + piece.Position.Y * 0.61f);
            Pieces.push_back(std::move(piece));
        }
        CreateFragmentBodies();
        Fractured = true;
        return true;
    }

    void DestructionSet2D::CreateFragmentBodies()
    {
        if (World == nullptr) return;
        for (DestructionPiece2D& piece : Pieces)
        {
            RigidBodyDesc2D description;
            description.Shape = std::make_shared<Convex2>(piece.LocalVertices);
            description.Transform.Position = piece.Position;
            description.Transform.Angle = piece.Angle;
            description.LinearVelocity = piece.Velocity;
            description.AngularVelocity = piece.AngularVelocity;
            description.Density = Settings.FragmentDensity;
            description.Restitution = Settings.Restitution;
            description.LinearDamping = Settings.LinearDamping;
            piece.Body = &World->CreateBody(description);
        }
    }

    void DestructionSet2D::Step(float deltaTime)
    {
        if (!Fractured || deltaTime <= 0.0f) return;
        deltaTime = std::min(deltaTime, 1.0f / 30.0f);
        const float damping = std::exp(-Settings.LinearDamping * deltaTime);
        for (DestructionPiece2D& piece : Pieces)
        {
            if (piece.Body != nullptr)
            {
                piece.Position = piece.Body->GetTransform().Position;
                piece.Angle = piece.Body->GetTransform().Angle;
                piece.Velocity = piece.Body->GetLinearVelocity();
                piece.AngularVelocity = piece.Body->GetAngularVelocity();
                continue;
            }
            piece.Velocity = damping * (piece.Velocity + deltaTime * Settings.Gravity);
            piece.Position += deltaTime * piece.Velocity;
            piece.Angle += deltaTime * piece.AngularVelocity;
            float minimumY = piece.Position.Y;
            for (const Vector2& local : piece.LocalVertices)
                minimumY = std::min(minimumY, piece.Position.Y + local.Rotated(piece.Angle).Y);
            if (minimumY < Settings.GroundHeight)
            {
                piece.Position.Y += Settings.GroundHeight - minimumY;
                if (piece.Velocity.Y < 0.0f) piece.Velocity.Y *= -Settings.Restitution;
                piece.Velocity.X *= 0.85f;
                piece.AngularVelocity *= 0.82f;
            }
        }
    }

    void DestructionSet2D::Clear()
    {
        IntactVertices.clear();
        Pieces.clear();
        World = nullptr;
        Fractured = false;
        TargetPieceCount = 1;
    }
}
