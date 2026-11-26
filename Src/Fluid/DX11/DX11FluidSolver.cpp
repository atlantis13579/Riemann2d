#if defined(_WIN32) && defined(RIEMANN2D_HAS_DX11)
#include "Fluid/GpuFluidSolvers.h"
#include "Compute/DX11Compute.h"

#include "Fluid/FluidSimulation2D.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Riemann2D
{
    namespace
    {
        using namespace DX11Compute;
        const char *FluidShaders =
#include "FluidDX11Kernels.h"
            ;
        class DX11FluidSolver final : public FluidGpuSolver
        {
            std::unique_ptr<DX11Device> Queue;
            SharedBuffer Particles, Scratch, Previous, Lambdas, Heads, Next;
            SharedProgram ClearHeads, Predict, BuildGrid, DensityLambda, Correct, Reconstruct, Viscosity;
            std::vector<FluidParticle2D> Readback;

          public:
            bool Step(const FluidSettings2D &settings, float restDensity, float deltaTime, int substeps,
                      std::vector<FluidParticle2D> &particles, std::string &error) override
            {
                try
                {
                    if (!Queue)
                    {
                        Queue = CreateDX11Device();
                        if (!Queue)
                            throw std::runtime_error("Compute Shader support was not built");
                        ClearHeads = Queue->CreateProgram(FluidShaders, "ClearHeads");
                        Predict = Queue->CreateProgram(FluidShaders, "Predict");
                        BuildGrid = Queue->CreateProgram(FluidShaders, "BuildGrid");
                        DensityLambda = Queue->CreateProgram(FluidShaders, "DensityLambda");
                        Correct = Queue->CreateProgram(FluidShaders, "Correct");
                        Reconstruct = Queue->CreateProgram(FluidShaders, "Reconstruct");
                        Viscosity = Queue->CreateProgram(FluidShaders, "Viscosity");
                    }
                    if (particles.empty())
                        return true;
                    if (particles.size() > 4000000u)
                        throw std::runtime_error("Compute liquid particle count exceeds capacity");
                    const int count = static_cast<int>(particles.size());
                    const float radius = std::max(settings.SmoothingRadius, 2.0f * settings.ParticleRadius);
                    if (!(radius > 0.0f))
                        throw std::runtime_error("Invalid Compute liquid smoothing radius");
                    Vector2 minimum = Vector2::InfMax(), maximum = Vector2::InfMin();
                    for (float x : {settings.BoundsMinimum.X, settings.BoundsMaximum.X})
                        for (float y : {settings.BoundsMinimum.Y, settings.BoundsMaximum.Y})
                        {
                            const Vector2 corner = settings.ContainerTransform.TransformPoint({x, y});
                            minimum = minimum.Min(corner);
                            maximum = maximum.Max(corner);
                        }
                    minimum -= radius;
                    const double width = std::ceil((maximum.X - minimum.X) / radius) + 2.0;
                    const double height = std::ceil((maximum.Y - minimum.Y) / radius) + 2.0;
                    if (!std::isfinite(width * height) || width * height > 4000000.0)
                        throw std::runtime_error("Compute liquid grid exceeds capacity");
                    const int cells = static_cast<int>(width) * static_cast<int>(height);
                    Constants p;
                    p.Integer(0, count);
                    p.Integer(1, static_cast<int>(width));
                    p.Integer(2, static_cast<int>(height));
                    p.Float(4, settings.BoundsMinimum.X);
                    p.Float(5, settings.BoundsMinimum.Y);
                    p.Float(6, settings.BoundsMaximum.X);
                    p.Float(7, settings.BoundsMaximum.Y);
                    p.Float(8, settings.ContainerTransform.Position.X);
                    p.Float(9, settings.ContainerTransform.Position.Y);
                    p.Float(10, settings.Gravity.X);
                    p.Float(11, settings.Gravity.Y);
                    p.Float(12, minimum.X);
                    p.Float(13, minimum.Y);
                    p.Float(14, std::cos(settings.ContainerTransform.Angle));
                    p.Float(15, std::sin(settings.ContainerTransform.Angle));
                    p.Float(16, radius);
                    p.Float(17, settings.ParticleRadius);
                    p.Float(18, 1.0f / restDensity);
                    p.Float(19, std::clamp(settings.PressureStiffness, 0.0f, 1.0f));
                    p.Float(20, 1.0f - std::exp(-std::max(settings.Viscosity, 0.0f) * deltaTime * 60.0f));
                    p.Float(21, settings.BoundaryRestitution);
                    p.Float(22, deltaTime);
                    static_assert(sizeof(FluidParticle2D) == 5 * sizeof(float), "GPU particle layout must match");
                    ReserveDX11Buffer(*Queue, Particles, count * sizeof(FluidParticle2D));
                    ReserveDX11Buffer(*Queue, Scratch, Particles->Size);
                    ReserveDX11Buffer(*Queue, Previous, count * 2 * sizeof(float));
                    ReserveDX11Buffer(*Queue, Lambdas, count * sizeof(float));
                    ReserveDX11Buffer(*Queue, Next, count * sizeof(int));
                    ReserveDX11Buffer(*Queue, Heads, cells * sizeof(int));
                    Readback.resize(count);
                    Queue->Upload(Particles, particles.data(), particles.size() * sizeof(FluidParticle2D));
                    const auto dispatch = [&](const SharedProgram &program, int threads)
                    { Queue->Dispatch(program, {Particles, Scratch, Heads, Next, Lambdas, Previous}, p, threads); };
                    const auto rebuild = [&]
                    {
                        dispatch(ClearHeads, cells);
                        dispatch(BuildGrid, count);
                    };
                    for (int step = 0; step < substeps; ++step)
                    {
                        dispatch(Predict, count);
                        for (int iteration = 0; iteration < std::max(settings.PressureIterations, 1); ++iteration)
                        {
                            rebuild();
                            dispatch(DensityLambda, count);
                            dispatch(Correct, count);
                            std::swap(Particles, Scratch);
                        }
                        dispatch(Reconstruct, count);
                        rebuild();
                        dispatch(DensityLambda, count);
                        dispatch(Viscosity, count);
                        std::swap(Particles, Scratch);
                    }
                    Queue->Download({{Particles, Readback.data(), Readback.size() * sizeof(FluidParticle2D)}});
                    particles.swap(Readback);
                    error.clear();
                    return true;
                }
                catch (const std::exception &exception)
                {
                    if (Queue)
                        Queue->Finish();
                    error = exception.what();
                    return false;
                }
            }
        };
    } // namespace
    std::unique_ptr<FluidGpuSolver> CreateDX11FluidSolver()
    {
        return std::make_unique<DX11FluidSolver>();
    }
} // namespace Riemann2D

#endif
