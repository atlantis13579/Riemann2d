#include "Viewer.h"

#include "FluidSurfaceBuilder.h"
#include "GeometryBuilder.h"
#include "Render/RHI/PlatformRHI.h"
#include "Collision/Box2.h"
#include "Collision/Capsule2.h"
#include "Collision/Convex2.h"
#include "Collision/Sphere2.h"
#include "Collision/TriangleMesh2.h"
#include "Fluid/FluidRigidCoupling2D.h"

#include <cmath>
#include <memory>

namespace Riemann2D::Viewer
{
    bool Viewer::Initialize(void* nativeWindow, int width, int height)
    {
        Width = width;
        Height = height;
        Scenes = {{"SHAPE GALLERY"}, {"JOINT CHAIN"},
                  {"ROPE XPBD"}, {"FLUID GRAVITY"}, {"AIR NO GRAVITY"},
                  {"DESTRUCTION"}, {"RAGDOLL"}};
        RebuildPhysicsScene();

        Render::NativeWindow window;
#if defined(_WIN32)
        window.Platform = Render::WindowPlatform::Win32;
#elif defined(__APPLE__)
        window.Platform = Render::WindowPlatform::Cocoa;
#endif
        window.Handle = nativeWindow;
        window.Width = width;
        window.Height = height;
        return Renderer.Initialize([] { return Render::CreatePlatformRHI(); }, window);
    }

    void Viewer::Resize(int width, int height)
    {
        if (width > 0 && height > 0)
        {
            Width = width;
            Height = height;
            Renderer.Resize(width, height);
        }
    }

    void Viewer::Tick(float deltaTime, const Render::InterfaceInput& input)
    {
        AnimationTime += deltaTime;
        HandleFluidInteraction(input);
        ConfineFluidBodiesToTank();
        World.SetGravity({0.0f, Gravity});
        if (Fluid) Fluid->GetSettings().Gravity = {0.0f, Gravity};
        World.SetContactSolver(SelectedSolver == 0 ? ContactSolverType2D::SequentialImpulse
                                                   : ContactSolverType2D::LCP,
                               SelectedBackend == 0 ? SolverBackend2D::ScalarCPU
                                                    : SolverBackend2D::SIMDCPU);
        World.Step(deltaTime);
        ConfineFluidBodiesToTank();
        if (Rope)
        {
            Rope->GetSettings().Gravity = {0.0f, Gravity};
            Rope->Step(deltaTime);
        }
        if (Fluid)
        {
            Fluid->Step(deltaTime);
        }
        if (Air)
        {
            Air->AddDensity({-5.5f, 3.0f}, 0.25f);
            Air->AddVelocity({-5.5f, 3.0f}, {0.8f, 0.25f * std::sin(AnimationTime)});
            Air->Step(deltaTime);
        }
        if (Destruction)
        {
            if (DestructionProjectile && !Destruction->IsFractured() &&
                DestructionProjectile->GetTransform().Position.Y < 5.7f)
                Destruction->Fracture({0.0f, 5.0f}, 4.0f);
            Destruction->Step(deltaTime);
        }
        std::shared_ptr<Render::FramePacket> frame = BuildFrame(input);
        LastSubmittedFrame = frame->FrameId;
        Renderer.Submit(std::move(frame));
    }

    std::shared_ptr<Render::FramePacket> Viewer::BuildFrame(const Render::InterfaceInput& input)
    {
        auto frame = std::make_shared<Render::FramePacket>();
        frame->FrameId = NextFrameId++;
        frame->ViewportWidth = Width;
        frame->ViewportHeight = Height;
        frame->PanelWidth = 260;
        frame->Camera.Center = {0.0f, 4.0f};
        frame->Camera.VerticalSize = 16.0f;

        Interface.Begin(Width, Height, frame->PanelWidth, input);
        Interface.Label("RIEMANN 2D", {0.35f, 0.75f, 1.0f, 1.0f});
        Interface.Label("SCENES");
        for (std::size_t index = 0; index < Scenes.size(); ++index)
        {
            if (Interface.Button(Scenes[index].Name, SelectedScene == static_cast<int>(index)))
            {
                SelectedScene = static_cast<int>(index);
                AnimationTime = 0.0f;
                RebuildPhysicsScene();
            }
        }
        Interface.Label("PARAMETERS");
        if (SelectedScene == FluidScene)
        {
            Interface.Label("LMB BODY: DRAG", {0.65f, 0.78f, 0.92f, 1.0f});
            Interface.Label("LMB EMPTY: ROTATE TANK", {0.65f, 0.78f, 0.92f, 1.0f});
            if (Interface.Button(RenderFluidSurface
                                     ? "FLUID RENDER: WATER"
                                     : "FLUID RENDER: PARTICLES",
                                 true))
            {
                RenderFluidSurface = !RenderFluidSurface;
            }
        }
        Interface.Slider("GRAVITY", Gravity, -20.0f, 0.0f);
        Interface.Label("SOLVER");
        if (Interface.Button("SEQUENTIAL", SelectedSolver == 0)) SelectedSolver = 0;
        if (Interface.Button("LCP", SelectedSolver == 1)) SelectedSolver = 1;
        Interface.Label("BACKEND");
        if (Interface.Button("SCALAR CPU", SelectedBackend == 0)) SelectedBackend = 0;
        if (Interface.Button("SIMD CPU", SelectedBackend == 1)) SelectedBackend = 1;
        Interface.Label("DX11 METAL RHI", {0.48f, 0.55f, 0.68f, 1.0f});

        frame->WorldDraws.push_back(BuildGridDraw());
        if (Fluid && RenderFluidSurface)
        {
            FluidSurfaceDraw surface = BuildFluidSurfaceDraw(*Fluid);
            frame->WorldDraws.push_back(std::move(surface.Fill));
            frame->WorldDraws.push_back(std::move(surface.Outline));
        }
        else if (Fluid)
        {
            Render::DrawPacket particles;
            particles.DebugName = "Fluid particles";
            particles.Topology = Render::PrimitiveTopology::Triangles;
            for (const FluidParticle2D& particle : Fluid->GetParticles())
            {
                const float shade = Clamp(0.35f + 0.08f * particle.Density, 0.35f, 0.9f);
                AddSolidCircle(particles, particle.Position, Fluid->GetSettings().ParticleRadius,
                               {0.08f, 0.35f + 0.35f * shade, shade, 0.95f}, 10);
            }
            frame->WorldDraws.push_back(std::move(particles));
        }
        frame->WorldDraws.push_back(BuildWorldDraw());
        frame->UserInterfaceDraws.push_back(Interface.End());
        return frame;
    }

    Render::DrawPacket Viewer::BuildWorldDraw() const
    {
        Render::DrawPacket solid;
        solid.DebugName = "Scene solids";
        solid.Topology = Render::PrimitiveTopology::Triangles;

        for (const std::unique_ptr<RigidBody2D>& body : World.GetBodies())
        {
            bool renderedAsDestructionPiece = false;
            if (Destruction != nullptr && Destruction->IsFractured())
            {
                for (const DestructionPiece2D& piece : Destruction->GetPieces())
                {
                    renderedAsDestructionPiece = renderedAsDestructionPiece ||
                        piece.Body == body.get();
                }
            }
            if (renderedAsDestructionPiece) continue;

            const Render::Color color = body->GetType() == BodyType2D::Static
                ? Render::Color{0.18f, 0.22f, 0.3f, 1.0f}
                : SelectedScene == FluidScene
                    ? Render::Color{0.95f, 0.55f, 0.18f, 1.0f}
                    : Render::Color{0.18f, 0.62f, 0.9f, 1.0f};
            if (body->GetShape().GetType() == ShapeType2::Box)
            {
                const auto& box = static_cast<const Box2&>(body->GetShape());
                AddSolidBox(solid, body->GetTransform().Position, box.GetHalfExtent(),
                            body->GetTransform().Angle, color);
            }
            else if (body->GetShape().GetType() == ShapeType2::Sphere2)
            {
                const auto& sphere = static_cast<const Sphere2&>(body->GetShape());
                AddSolidCircle(solid, body->GetTransform().Position, sphere.GetRadius(), color);
            }
            else if (body->GetShape().GetType() == ShapeType2::Capsule2)
            {
                const auto& capsule = static_cast<const Capsule2&>(body->GetShape());
                AddSolidCapsule(solid, body->GetTransform().Position, capsule.GetHalfLength(),
                                capsule.GetRadius(), body->GetTransform().Angle, color);
            }
            else if (body->GetShape().GetType() == ShapeType2::Convex2)
            {
                const auto& convex = static_cast<const Convex2&>(body->GetShape());
                AddSolidPolygon(solid, convex.GetVertices(), body->GetTransform().Position,
                                body->GetTransform().Angle, color);
            }
            else if (body->GetShape().GetType() == ShapeType2::TriangleMesh2)
            {
                const auto& mesh = static_cast<const TriangleMesh2&>(body->GetShape());
                for (std::size_t triangleIndex = 0;
                     triangleIndex < mesh.GetTriangleCount(); ++triangleIndex)
                {
                    const TriangleMesh2::Triangle triangle = mesh.GetTriangle(triangleIndex);
                    AddSolidPolygon(solid, {triangle[0], triangle[1], triangle[2]},
                                    body->GetTransform().Position, body->GetTransform().Angle,
                                    color);
                }
            }
        }

        if (Air)
        {
            const Vector2 halfCell = 0.5f * Vector2{
                (Air->GetBoundsMaximum().X - Air->GetBoundsMinimum().X) / static_cast<float>(Air->GetWidth()),
                (Air->GetBoundsMaximum().Y - Air->GetBoundsMinimum().Y) / static_cast<float>(Air->GetHeight())};
            for (int y = 0; y < Air->GetHeight(); ++y)
                for (int x = 0; x < Air->GetWidth(); ++x)
                {
                    const float density = Clamp(Air->GetDensity(x, y), 0.0f, 1.0f);
                    if (density > 0.015f)
                        AddSolidBox(solid, Air->GetCellCenter(x, y), halfCell, 0.0f,
                                    {0.25f, 0.7f, 0.9f, 0.5f * density});
                }
        }
        if (Destruction)
        {
            if (!Destruction->IsFractured())
                AddSolidPolygon(solid, Destruction->GetIntactVertices(), Vector2::Zero(), 0.0f,
                                {0.5f, 0.58f, 0.68f, 1.0f});
            else
                for (const DestructionPiece2D& piece : Destruction->GetPieces())
                    AddSolidPolygon(solid, piece.LocalVertices, piece.Position, piece.Angle,
                                    {0.62f, 0.4f, 0.24f, 1.0f});
        }

        return solid;
    }

    Render::DrawPacket Viewer::BuildGridDraw() const
    {
        Render::DrawPacket grid;
        grid.DebugName = "XY grid";
        grid.Topology = Render::PrimitiveTopology::Lines;
        const Render::Color gridColor{0.09f, 0.12f, 0.17f, 1.0f};
        for (int coordinate = -10; coordinate <= 10; ++coordinate)
        {
            AddLine(grid, {static_cast<float>(coordinate), -4.0f}, {static_cast<float>(coordinate), 12.0f}, gridColor);
            AddLine(grid, {-10.0f, static_cast<float>(coordinate)}, {10.0f, static_cast<float>(coordinate)}, gridColor);
        }
        if (Rope)
        {
            const std::vector<RopeParticle2D>& particles = Rope->GetParticles();
            for (std::size_t index = 0; index + 1 < particles.size(); ++index)
            {
                AddLine(grid, particles[index].Position, particles[index + 1].Position,
                        {0.25f, 0.8f, 1.0f, 1.0f});
            }
        }
        if (Air)
        {
            for (int y = 1; y < Air->GetHeight(); y += 3)
                for (int x = 1; x < Air->GetWidth(); x += 3)
                {
                    const Vector2 center = Air->GetCellCenter(x, y);
                    AddLine(grid, center, center + 0.18f * Air->GetVelocity(x, y),
                            {0.5f, 0.85f, 1.0f, 0.7f});
                }
        }

        return grid;
    }

    Vector2 Viewer::ScreenToWorld(const Render::InterfaceInput& input) const
    {
        constexpr int panelWidth = 260;
        constexpr float verticalSize = 16.0f;
        const float viewportHeight = static_cast<float>(std::max(Height, 1));
        const float worldWidth = static_cast<float>(std::max(Width - panelWidth, 1));
        const float horizontalSize = verticalSize * worldWidth / viewportHeight;
        const float localX = 2.0f *
            (static_cast<float>(input.MouseX - panelWidth) / worldWidth) - 1.0f;
        const float localY = 1.0f - 2.0f *
            (static_cast<float>(input.MouseY) / viewportHeight);
        return {0.5f * horizontalSize * localX,
                4.0f + 0.5f * verticalSize * localY};
    }

    RigidBody2D* Viewer::PickFluidBody(const Vector2& worldPosition) const
    {
        for (auto body = FluidBodies.rbegin(); body != FluidBodies.rend(); ++body)
        {
            const Vector2 localPosition =
                (*body)->GetTransform().InverseTransformPoint(worldPosition);
            if ((*body)->GetShape().GetType() == ShapeType2::Box)
            {
                const Vector2 halfExtent =
                    static_cast<const Box2&>((*body)->GetShape()).GetHalfExtent();
                if (std::abs(localPosition.X) <= halfExtent.X &&
                    std::abs(localPosition.Y) <= halfExtent.Y)
                {
                    return *body;
                }
            }
            else if ((*body)->GetShape().GetType() == ShapeType2::Sphere2)
            {
                const float radius =
                    static_cast<const Sphere2&>((*body)->GetShape()).GetRadius();
                if (localPosition.SquareLength() <= radius * radius) return *body;
            }
        }
        return nullptr;
    }

    void Viewer::HandleFluidInteraction(const Render::InterfaceInput& input)
    {
        if (SelectedScene != FluidScene || Fluid == nullptr)
        {
            CancelFluidInteraction();
            return;
        }
        if (!input.MouseDown)
        {
            CancelFluidInteraction();
            return;
        }

        const Vector2 worldPosition = ScreenToWorld(input);
        if (input.MousePressed && input.MouseX > 260)
        {
            if (RigidBody2D* body = PickFluidBody(worldPosition))
            {
                MouseConstraintDesc2D description;
                description.Body = body;
                description.LocalAnchor =
                    body->GetTransform().InverseTransformPoint(worldPosition);
                description.Target = worldPosition;
                description.Frequency = 4.0f;
                description.Damping = 0.9f;
                description.MaxForce = 600.0f;
                auto constraint = std::make_unique<MouseConstraint2D>(description);
                ActiveMouseConstraint = constraint.get();
                World.AddConstraint(std::move(constraint));
            }
            else
            {
                const Vector2 offset = worldPosition - FluidTankTransform.Position;
                if (offset.SquareLength() > 0.04f)
                {
                    RotatingFluidTank = true;
                    FluidTankStartAngle = FluidTankTransform.Angle;
                    MouseStartAngle = std::atan2(offset.Y, offset.X);
                }
            }
        }

        if (ActiveMouseConstraint != nullptr)
        {
            ActiveMouseConstraint->SetTarget(worldPosition);
        }
        else if (RotatingFluidTank)
        {
            const Vector2 offset = worldPosition - FluidTankTransform.Position;
            if (offset.SquareLength() > 0.04f)
            {
                FluidTankTransform.Angle = FluidTankStartAngle +
                    std::remainder(std::atan2(offset.Y, offset.X) - MouseStartAngle,
                                   2.0f * Pi);
                UpdateFluidTankTransform();
            }
        }
    }

    void Viewer::CancelFluidInteraction()
    {
        if (ActiveMouseConstraint != nullptr)
        {
            World.RemoveConstraint(ActiveMouseConstraint);
            ActiveMouseConstraint = nullptr;
        }
        RotatingFluidTank = false;
    }

    void Viewer::UpdateFluidTankTransform()
    {
        if (Fluid != nullptr)
        {
            Fluid->GetSettings().ContainerTransform = FluidTankTransform;
        }
        for (const FluidTankWall& wall : FluidTankWalls)
        {
            Transform2 transform;
            transform.Position =
                FluidTankTransform.TransformPoint(wall.LocalTransform.Position);
            transform.Angle = FluidTankTransform.Angle + wall.LocalTransform.Angle;
            wall.Body->SetTransform(transform);
        }
    }

    void Viewer::ConfineFluidBodiesToTank()
    {
        if (Fluid == nullptr) return;
        const FluidSettings2D& settings = Fluid->GetSettings();
        for (RigidBody2D* body : FluidBodies)
        {
            float radius = 0.0f;
            if (body->GetShape().GetType() == ShapeType2::Box)
            {
                radius = static_cast<const Box2&>(body->GetShape()).GetHalfExtent().Length();
            }
            else if (body->GetShape().GetType() == ShapeType2::Sphere2)
            {
                radius = static_cast<const Sphere2&>(body->GetShape()).GetRadius();
            }
            const Vector2 minimum = settings.BoundsMinimum + Vector2{radius, radius};
            const Vector2 maximum = settings.BoundsMaximum - Vector2{radius, radius};
            Transform2 transform = body->GetTransform();
            Vector2 localPosition =
                FluidTankTransform.InverseTransformPoint(transform.Position);
            const Vector2 originalPosition = localPosition;
            localPosition.X = Clamp(localPosition.X, minimum.X, maximum.X);
            localPosition.Y = Clamp(localPosition.Y, minimum.Y, maximum.Y);
            if ((localPosition - originalPosition).SquareLength() <= Epsilon) continue;

            Vector2 localVelocity =
                FluidTankTransform.InverseTransformVector(body->GetLinearVelocity());
            if ((originalPosition.X < minimum.X && localVelocity.X < 0.0f) ||
                (originalPosition.X > maximum.X && localVelocity.X > 0.0f))
            {
                localVelocity.X = 0.0f;
            }
            if ((originalPosition.Y < minimum.Y && localVelocity.Y < 0.0f) ||
                (originalPosition.Y > maximum.Y && localVelocity.Y > 0.0f))
            {
                localVelocity.Y = 0.0f;
            }
            transform.Position = FluidTankTransform.TransformPoint(localPosition);
            body->SetTransform(transform);
            body->SetLinearVelocity(
                FluidTankTransform.TransformVector(localVelocity));
        }
    }

    void Viewer::RebuildPhysicsScene()
    {
        CancelFluidInteraction();
        Ragdoll.reset();
        Destruction.reset();
        World.Clear();
        Rope.reset();
        Fluid.reset();
        FluidTankWalls.clear();
        FluidBodies.clear();
        FluidTankTransform = {};
        Air.reset();
        DestructionProjectile = nullptr;
        if (SelectedScene == ShapeGalleryScene)
        {
            RigidBodyDesc2D mesh;
            mesh.Type = BodyType2D::Static;
            mesh.Shape = std::make_shared<TriangleMesh2>(
                std::vector<Vector2>{{-8.0f, -0.35f}, {8.0f, -0.35f},
                                     {8.0f, 0.35f}, {-8.0f, 0.35f}},
                std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3});
            mesh.Transform.Position = {0.0f, -1.0f};
            World.CreateBody(mesh);

            RigidBodyDesc2D body;
            body.Shape = std::make_shared<Box2>(Vector2{0.9f, 0.9f});
            body.Transform.Position = {-4.8f, 7.0f};
            body.Transform.Angle = 0.25f;
            World.CreateBody(body);

            body = {};
            body.Shape = std::make_shared<Sphere2>(0.9f);
            body.Transform.Position = {-2.2f, 8.0f};
            World.CreateBody(body);

            body = {};
            body.Shape = std::make_shared<Capsule2>(1.0f, 0.5f);
            body.Transform.Position = {0.4f, 7.5f};
            body.Transform.Angle = 0.4f;
            World.CreateBody(body);

            body = {};
            body.Shape = std::make_shared<Convex2>(std::vector<Vector2>{
                {-1.0f, -0.45f}, {-0.2f, -1.0f}, {0.9f, -0.55f},
                {1.0f, 0.5f}, {0.0f, 1.0f}, {-0.85f, 0.55f}});
            body.Transform.Position = {3.4f, 8.2f};
            body.Transform.Angle = -0.15f;
            World.CreateBody(body);
            return;
        }

        if (SelectedScene == JointChainScene)
        {
            RigidBodyDesc2D anchorDescription;
            anchorDescription.Type = BodyType2D::Static;
            anchorDescription.Shape = std::make_shared<Box2>(Vector2{0.2f, 0.15f});
            anchorDescription.Transform.Position = {-3.0f, 8.15f};
            RigidBody2D* previous = &World.CreateBody(anchorDescription);
            for (int linkIndex = 0; linkIndex < 6; ++linkIndex)
            {
                RigidBodyDesc2D linkDescription;
                linkDescription.Shape = std::make_shared<Box2>(Vector2{0.28f, 0.65f});
                linkDescription.Transform.Position = {-3.0f, 7.35f - 1.3f * static_cast<float>(linkIndex)};
                RigidBody2D* link = &World.CreateBody(linkDescription);
                RevoluteConstraintDesc2D joint;
                joint.BodyA = previous;
                joint.BodyB = link;
                joint.LocalAnchorA = linkIndex == 0 ? Vector2{0.0f, -0.15f} : Vector2{0.0f, -0.65f};
                joint.LocalAnchorB = {0.0f, 0.65f};
                joint.EnableLimit = true;
                joint.LowerAngle = -0.7f;
                joint.UpperAngle = 0.7f;
                World.AddConstraint(std::make_unique<RevoluteConstraint2D>(joint));
                previous = link;
            }

            anchorDescription.Transform.Position = {4.0f, 5.0f};
            RigidBody2D& rail = World.CreateBody(anchorDescription);
            RigidBodyDesc2D sliderDescription;
            sliderDescription.Shape = std::make_shared<Box2>(Vector2{0.65f, 0.4f});
            sliderDescription.Transform.Position = {4.0f, 2.5f};
            RigidBody2D& slider = World.CreateBody(sliderDescription);
            PrismaticConstraintDesc2D sliderJoint;
            sliderJoint.BodyA = &rail;
            sliderJoint.BodyB = &slider;
            sliderJoint.LocalAxisA = Vector2::UnitY();
            sliderJoint.EnableLimit = true;
            sliderJoint.LowerTranslation = -3.5f;
            sliderJoint.UpperTranslation = -0.8f;
            sliderJoint.EnableMotor = true;
            sliderJoint.MotorSpeed = 1.0f;
            sliderJoint.MaxMotorForce = 30.0f;
            World.AddConstraint(std::make_unique<PrismaticConstraint2D>(sliderJoint));
            return;
        }

        if (SelectedScene == RopeScene)
        {
            Rope = std::make_unique<RopeSimulation2D>();
            Rope->CreateLine({-6.0f, 7.0f}, {6.0f, 7.0f}, 31, 0.5f);
            Rope->PinParticle(0, {-6.0f, 7.0f});
            Rope->PinParticle(30, {6.0f, 7.0f});
            return;
        }
        if (SelectedScene == FluidScene)
        {
            Fluid = std::make_unique<FluidSimulation2D>();
            FluidTankTransform.Position = {0.0f, 3.5f};
            Fluid->GetSettings().BoundsMinimum = {-6.0f, -5.0f};
            Fluid->GetSettings().BoundsMaximum = {6.0f, 5.0f};
            Fluid->GetSettings().ContainerTransform = FluidTankTransform;
            Fluid->GetSettings().SmoothingRadius = 0.42f;
            Fluid->GetSettings().ParticleRadius = 0.04f;
            Fluid->CreateBlock({-5.5f, -4.5f}, {5.5f, 0.1f}, 0.38f);

            RigidBodyDesc2D wall;
            wall.Type = BodyType2D::Static;
            wall.Shape = std::make_shared<Box2>(Vector2{6.5f, 0.25f});
            wall.Transform.Position = FluidTankTransform.TransformPoint({0.0f, -5.25f});
            FluidTankWalls.push_back({&World.CreateBody(wall), {{0.0f, -5.25f}, 0.0f}});
            wall.Transform.Position = FluidTankTransform.TransformPoint({0.0f, 5.25f});
            FluidTankWalls.push_back({&World.CreateBody(wall), {{0.0f, 5.25f}, 0.0f}});
            wall.Shape = std::make_shared<Box2>(Vector2{0.25f, 5.0f});
            wall.Transform.Position = FluidTankTransform.TransformPoint({-6.25f, 0.0f});
            FluidTankWalls.push_back({&World.CreateBody(wall), {{-6.25f, 0.0f}, 0.0f}});
            wall.Transform.Position = FluidTankTransform.TransformPoint({6.25f, 0.0f});
            FluidTankWalls.push_back({&World.CreateBody(wall), {{6.25f, 0.0f}, 0.0f}});

            RigidBodyDesc2D floatingBody;
            floatingBody.Density = 0.55f;
            floatingBody.LinearDamping = 0.3f;
            floatingBody.AngularDamping = 0.4f;
            floatingBody.Shape = std::make_shared<Box2>(Vector2{0.7f, 0.45f});
            floatingBody.Transform.Position =
                FluidTankTransform.TransformPoint({-2.7f, 0.5f});
            floatingBody.Transform.Angle = 0.18f;
            FluidBodies.push_back(&World.CreateBody(floatingBody));
            floatingBody.Shape = std::make_shared<Box2>(Vector2{0.5f, 0.75f});
            floatingBody.Transform.Position =
                FluidTankTransform.TransformPoint({1.2f, 0.8f});
            floatingBody.Transform.Angle = -0.25f;
            FluidBodies.push_back(&World.CreateBody(floatingBody));
            floatingBody.Density = 0.65f;
            floatingBody.Shape = std::make_shared<Sphere2>(0.65f);
            floatingBody.Transform.Position =
                FluidTankTransform.TransformPoint({-0.5f, 1.1f});
            floatingBody.Transform.Angle = 0.0f;
            FluidBodies.push_back(&World.CreateBody(floatingBody));
            floatingBody.Shape = std::make_shared<Sphere2>(0.5f);
            floatingBody.Transform.Position =
                FluidTankTransform.TransformPoint({3.4f, 0.4f});
            FluidBodies.push_back(&World.CreateBody(floatingBody));

            World.AddForceField(std::make_unique<FluidRigidCoupling2D>(Fluid.get()));
            UpdateFluidTankTransform();
            return;
        }
        if (SelectedScene == AirScene)
        {
            Air = std::make_unique<AerodynamicsSimulation2D>();
            Air->Initialize({-7.0f, -2.0f}, {7.0f, 9.0f}, 480, 360);
            return;
        }
        if (SelectedScene == DestructionScene)
        {
            RigidBodyDesc2D ground;
            ground.Type = BodyType2D::Static;
            ground.Shape = std::make_shared<Box2>(Vector2{8.0f, 0.35f});
            ground.Transform.Position = {0.0f, -1.0f};
            World.CreateBody(ground);
            RigidBodyDesc2D projectile;
            projectile.Shape = std::make_shared<Box2>(Vector2{1.25f, 1.25f});
            projectile.Transform.Position = {0.0f, 10.0f};
            projectile.Density = 4.0f;
            DestructionProjectile = &World.CreateBody(projectile);
            Destruction = std::make_unique<DestructionSet2D>();
            Destruction->CreateBox({0.0f, 2.0f}, {2.0f, 3.0f}, 4);
            Destruction->SetPhysicsWorld(&World);
            return;
        }
        if (SelectedScene == RagdollScene)
        {
            RigidBodyDesc2D ground;
            ground.Type = BodyType2D::Static;
            ground.Shape = std::make_shared<Box2>(Vector2{8.0f, 0.35f});
            ground.Transform.Position = {0.0f, -1.0f};
            World.CreateBody(ground);
            Ragdoll = std::make_unique<Ragdoll2D>();
            Ragdoll->Build(World, {0.0f, 5.5f}, 0.9f);
            return;
        }

    }

    void Viewer::Shutdown()
    {
        Renderer.WaitIdle(LastSubmittedFrame);
        Renderer.Shutdown();
    }
}
