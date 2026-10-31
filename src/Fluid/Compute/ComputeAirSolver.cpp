#include "ComputeSolvers.h"
#include "ComputeDevice.h"
#include "AirComputeShaders.h"
#include "Fluid/AerodynamicsSimulation2D.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Riemann2D
{
    namespace
    {
        using namespace Compute;
        struct Level
        {
            int Width, Height;
            float WeightX, WeightY;
            SharedBuffer Pressure, RHS, Residual, InverseDiagonal, PartialSums, Mean;
        };
        enum Kernel
        {
            AdvectFaces, ComputeCurl, ComputeForce, ApplyForce, PreparePressure, Smooth,
            SmoothCoarsest, Residual, Restrict, ReduceMean, RemoveMean, Prolong, ApplyPressure,
            ForwardDensity, CorrectDensity, KernelCount
        };
        class AirSolver final : public AirGpuSolver
        {
            std::unique_ptr<Device> Queue;
            std::array<SharedProgram, KernelCount> Programs;
            SharedBuffer U, V, UScratch, VScratch, Density, Forward, DensityScratch, Curl, Centers, Force;
            std::vector<Level> Levels;
            std::vector<float> ReadU, ReadV, ReadDensity;
            int Width = 0, Height = 0;
            Vector2 CellSize;

            void Dispatch(Kernel kernel, const Bindings& bindings, const Constants& p, int count)
            {
                Queue->Dispatch(Programs[kernel], bindings, p, count);
            }
            void Cycle(std::size_t index)
            {
                auto& level = Levels[index];
                const int count = level.Width * level.Height;
                Constants p; p.Integer(16, level.Width); p.Integer(17, level.Height);
                p.Float(18, level.WeightX); p.Float(19, level.WeightY);
                const Bindings bindings{level.Pressure, level.RHS, level.InverseDiagonal, level.Residual};
                if (index + 1 == Levels.size()) { Dispatch(SmoothCoarsest, bindings, p, 256); return; }
                const auto smooth = [&] {
                    for (int iteration = 0; iteration < 2; ++iteration)
                        for (int parity = 0; parity < 2; ++parity)
                        { p.Integer(3, parity); Dispatch(Smooth, bindings, p, count); }
                };
                smooth(); Dispatch(Residual, bindings, p, count);
                auto& coarse = Levels[index + 1]; const int coarseCount = coarse.Width * coarse.Height;
                p.Integer(20, coarse.Width); p.Integer(21, coarse.Height);
                Dispatch(Restrict, {level.Residual, coarse.Pressure, coarse.RHS, coarse.PartialSums}, p, coarseCount);
                Dispatch(ReduceMean, {coarse.PartialSums, coarse.Mean}, p, 256);
                Dispatch(RemoveMean, {coarse.RHS, coarse.Mean}, p, coarseCount);
                Cycle(index + 1);
                Dispatch(Prolong, {level.Pressure, coarse.Pressure}, p, count); smooth();
            }
            void Resize(int width, int height, Vector2 cellSize)
            {
                if (width < 3 || height < 3 || static_cast<std::int64_t>(width) * height > 4000000)
                    throw std::runtime_error("Compute smoke grid exceeds capacity");
                Width = width; Height = height; CellSize = cellSize; Levels.clear();
                const int n = width * height, nu = (width + 1) * height, nv = width * (height + 1);
                Reserve(*Queue, U, nu * sizeof(float)); Reserve(*Queue, V, nv * sizeof(float));
                Reserve(*Queue, UScratch, nu * sizeof(float)); Reserve(*Queue, VScratch, nv * sizeof(float));
                for (auto* buffer : {&Density, &Forward, &DensityScratch, &Curl}) Reserve(*Queue, *buffer, n * sizeof(float));
                Reserve(*Queue, Centers, 2 * n * sizeof(float)); Reserve(*Queue, Force, 2 * n * sizeof(float));
                ReadU.resize(nu); ReadV.resize(nv); ReadDensity.resize(n);
                const Vector2 extent{width * cellSize.X, height * cellSize.Y};
                for (;;)
                {
                    Level level; level.Width = width; level.Height = height;
                    const int count = width * height;
                    const float inverseX = width / extent.X, inverseY = height / extent.Y;
                    level.WeightX = inverseX * inverseX; level.WeightY = inverseY * inverseY;
                    for (auto* buffer : {&level.Pressure, &level.RHS, &level.Residual, &level.InverseDiagonal})
                        Reserve(*Queue, *buffer, count * sizeof(float));
                    Reserve(*Queue, level.PartialSums, ((count + 255) / 256) * sizeof(float));
                    Reserve(*Queue, level.Mean, sizeof(float));
                    std::vector<float> inverse(count);
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x)
                            inverse[y * width + x] = 1.0f / (((x > 0) + (x + 1 < width)) * level.WeightX +
                                ((y > 0) + (y + 1 < height)) * level.WeightY);
                    Queue->Upload(level.InverseDiagonal, inverse.data(), count * sizeof(float));
                    Levels.push_back(std::move(level));
                    if (width <= 4 && height <= 4) break;
                    width = (width + 1) / 2; height = (height + 1) / 2;
                }
            }
        public:
            bool Step(const AerodynamicsSettings2D& settings, const Vector2& cellSize,
                      int width, int height, float deltaTime, bool projectOnly, std::vector<float>& density,
                      std::vector<float>& u, std::vector<float>& v, std::string& error) override
            {
                try
                {
                    if (!Queue)
                    {
                        Queue = CreateDevice();
                        if (!Queue) throw std::runtime_error("Compute Shader support was not built");
                        const char* entries[] = {"AdvectFaces", "ComputeCurl", "ComputeForce", "ApplyForce", "PreparePressure",
                            "Smooth", "SmoothCoarsest", "Residual", "Restrict", "ReduceMean", "RemoveMean", "Prolong",
                            "ApplyPressure", "ForwardDensity", "CorrectDensity"};
                        for (int i = 0; i < KernelCount; ++i) Programs[i] = Queue->CreateProgram(AirShaders, entries[i]);
                    }
                    if (Width != width || Height != height || CellSize.X != cellSize.X || CellSize.Y != cellSize.Y)
                        Resize(width, height, cellSize);
                    Constants p; p.Integer(0, width); p.Integer(1, height);
                    p.Float(4, 1.0f / cellSize.X); p.Float(5, 1.0f / cellSize.Y); p.Float(6, deltaTime);
                    p.Float(7, std::max(settings.VelocityDiffusion, 0.0f)); p.Float(8, settings.VorticityStrength);
                    p.Float(9, std::exp(-std::max(settings.DensityDissipation, 0.0f) * deltaTime));
                    p.Integer(10, settings.UseMacCormack ? 1 : 0);
                    Queue->Upload(U, u.data(), u.size() * sizeof(float));
                    Queue->Upload(V, v.data(), v.size() * sizeof(float));
                    const int n = width * height, nu = (width + 1) * height, nv = width * (height + 1);
                    if (!projectOnly)
                    {
                        Queue->Upload(Density, density.data(), density.size() * sizeof(float));
                        p.Integer(2, 0); Dispatch(AdvectFaces, {U, V, UScratch}, p, nu);
                        p.Integer(2, 1); Dispatch(AdvectFaces, {U, V, VScratch}, p, nv);
                        std::swap(U, UScratch); std::swap(V, VScratch);
                        if (settings.VorticityStrength > 0.0f)
                        {
                            Dispatch(ComputeCurl, {U, V, Curl}, p, n); Dispatch(ComputeForce, {Curl, Force}, p, n);
                            Dispatch(ApplyForce, {U, V, Force}, p, std::max(nu, nv));
                        }
                    }
                    Dispatch(PreparePressure, {U, V, Levels[0].Pressure, Levels[0].RHS}, p, n);
                    for (int cycle = 0; cycle < std::max(1, (settings.PressureIterations + 7) / 8); ++cycle) Cycle(0);
                    Dispatch(ApplyPressure, {U, V, Levels[0].Pressure}, p, std::max(nu, nv));
                    std::vector<Readback> requests{{U, ReadU.data(), u.size() * sizeof(float)},
                        {V, ReadV.data(), v.size() * sizeof(float)}};
                    if (!projectOnly)
                    {
                        Dispatch(ForwardDensity, {Density, U, V, Forward, Centers}, p, n);
                        Dispatch(CorrectDensity, {Density, Forward, Centers, DensityScratch}, p, n);
                        std::swap(Density, DensityScratch);
                        requests.push_back({Density, ReadDensity.data(), density.size() * sizeof(float)});
                    }
                    Queue->Download(requests);
                    u.swap(ReadU); v.swap(ReadV); if (!projectOnly) density.swap(ReadDensity);
                    error.clear(); return true;
                }
                catch (const std::exception& exception)
                {
                    if (Queue) Queue->Finish();
                    error = exception.what(); return false;
                }
            }
        };
    }
    std::unique_ptr<AirGpuSolver> CreateComputeAirSolver() { return std::make_unique<AirSolver>(); }
}
