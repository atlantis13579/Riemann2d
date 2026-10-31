#include "Test.h"

#include "Render/RHI/Null/NullRHI.h"
#include "Render/ImmediateUI.h"
#include "Render/Projection.h"
#include "Render/Renderer2.h"
#include "FluidSurfaceBuilder.h"

using namespace Riemann2D;
using namespace Riemann2D::Render;

R2_TEST("Fluid render fields retain isolated droplets and transformed container bounds")
{
    FluidSimulation2D fluid;
    auto& settings = fluid.GetSettings();
    settings.BoundsMinimum = {-1.0f, -1.0f};
    settings.BoundsMaximum = {1.0f, 1.0f};
    settings.ContainerTransform = {{2.0f, 3.0f}, 0.6f};
    fluid.CreateBlock({0.0f, 0.0f}, {0.0f, 0.0f}, 0.2f);
    const auto draw = Viewer::BuildFluidSurfaceDraw(fluid).Fill;
    R2_REQUIRE(draw.Vertices.size() == 4 && draw.Indices.size() == 6);
    R2_REQUIRE(draw.Field && draw.Field->IsValid());
    R2_REQUIRE(*std::max_element(draw.Field->Values.begin(), draw.Field->Values.end()) > 0.8f);
    for (const auto& vertex : draw.Vertices)
    {
        const auto local = settings.ContainerTransform.InverseTransformPoint(vertex.Position);
        R2_REQUIRE(std::abs(local.X) < 1.00001f && std::abs(local.Y) < 1.00001f);
    }
}

R2_TEST("Smoke render packet owns an immutable density snapshot with constant geometry")
{
    AerodynamicsSimulation2D air;
    air.Initialize({-1.0f, -1.0f}, {1.0f, 1.0f}, 80, 60);
    air.AddEmitter({0.0f, 0.0f}, 0.5f, 3.0f, {1.0f, 0.0f}, 0.1f);
    const auto draw = Viewer::BuildSmokeDraw(air);
    R2_REQUIRE(draw.Vertices.size() == 4 && draw.Indices.size() == 6);
    R2_REQUIRE(draw.Field && draw.Field->IsValid());
    R2_REQUIRE(draw.Field->Values == air.GetDensityField());
    const auto snapshot = draw.Field->Values;
    air.Clear();
    R2_REQUIRE(draw.Field->Values == snapshot);
}

R2_TEST("Render thread owns RHI lifecycle and presents immutable frame")
{
    const std::shared_ptr<NullRHIStats> stats = std::make_shared<NullRHIStats>();
    Renderer2 renderer;
    NativeWindow window;
    window.Width = 640;
    window.Height = 480;

    R2_REQUIRE(renderer.Initialize([stats] { return CreateNullRHI(stats); }, window));
    R2_REQUIRE(stats->Initialized);

    std::shared_ptr<FramePacket> frame = std::make_shared<FramePacket>();
    frame->FrameId = 7;
    frame->WorldDraws.emplace_back();
    R2_REQUIRE(renderer.Submit(frame));
    renderer.WaitIdle(frame->FrameId);

    R2_REQUIRE(stats->PresentedFrame == 7);
    R2_REQUIRE(stats->DrawPacketCount == 1);

    renderer.Resize(800, 600);
    renderer.Shutdown();
    R2_REQUIRE(stats->ShutdownCalled);
}

R2_TEST("Orthographic projection reserves the interface panel")
{
    FramePacket frame;
    frame.ViewportWidth = 1000;
    frame.ViewportHeight = 500;
    frame.PanelWidth = 200;
    frame.Camera.Center = {0.0f, 0.0f};
    frame.Camera.VerticalSize = 10.0f;

    const Vector2 worldCenter = ProjectWorldToClip(Vector2::Zero(), frame);
    R2_REQUIRE(NearlyEqual(worldCenter.X, 0.2f));
    R2_REQUIRE(NearlyEqual(worldCenter.Y, 0.0f));
    const Vector2 interfaceOrigin = ProjectInterfaceToClip(Vector2::Zero(), frame);
    R2_REQUIRE(NearlyEqual(interfaceOrigin.X, -1.0f));
    R2_REQUIRE(NearlyEqual(interfaceOrigin.Y, 1.0f));
}

R2_TEST("Immediate interface reports a clicked scene button")
{
    ImmediateUI ui;
    InterfaceInput input;
    input.MouseX = 30;
    input.MouseY = 45;
    input.MousePressed = true;
    ui.Begin(800, 600, 260, input);
    R2_REQUIRE(ui.Button("SCENE ONE"));
    const DrawPacket draw = ui.End();
    R2_REQUIRE(!draw.Vertices.empty());
    R2_REQUIRE(!draw.Indices.empty());
}

R2_TEST("Immediate interface disabled backend cannot be selected")
{
    ImmediateUI ui;
    InterfaceInput input;
    input.MouseX = 30; input.MouseY = 30; input.MousePressed = true;
    ui.Begin(800, 600, 260, input);
    R2_REQUIRE(!ui.Button("CUDA UNAVAILABLE", false, false));
    R2_REQUIRE(!ui.End().Vertices.empty());
}
