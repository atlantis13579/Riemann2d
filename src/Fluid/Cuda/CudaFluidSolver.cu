#include "CudaSolvers.h"
#include "CudaCommon.cuh"
#include "Fluid/FluidSimulation2D.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Riemann2D
{
    namespace
    {
        using namespace Cuda;
        struct Particle { Float2 Position; Float2 Velocity; float Density; };
        static_assert(sizeof(Particle) == sizeof(FluidParticle2D), "Particle transfer layout must match");
        static_assert(offsetof(Particle, Density) == offsetof(FluidParticle2D, Density), "Particle offsets must match");
        struct Parameters
        {
            Float2 Minimum, Maximum, Center, Gravity, GridMinimum;
            float Cosine, Sine, Radius, ParticleRadius, InverseDensity, Stiffness, Viscosity, Restitution, TimeStep;
            int Count, GridWidth, GridHeight;
        };
        __device__ Float2 Rotate(Float2 v, float c, float s) { return {c * v.X - s * v.Y, s * v.X + c * v.Y}; }
        __device__ void Constrain(Particle& particle, const Parameters& p)
        {
            Float2 position = Rotate(particle.Position - p.Center, p.Cosine, -p.Sine);
            Float2 velocity = Rotate(particle.Velocity, p.Cosine, -p.Sine);
            if (position.X <= p.Minimum.X + p.ParticleRadius)
            { position.X = p.Minimum.X + p.ParticleRadius; if (velocity.X < 0.0f) velocity.X *= -p.Restitution; }
            if (position.X >= p.Maximum.X - p.ParticleRadius)
            { position.X = p.Maximum.X - p.ParticleRadius; if (velocity.X > 0.0f) velocity.X *= -p.Restitution; }
            if (position.Y <= p.Minimum.Y + p.ParticleRadius)
            { position.Y = p.Minimum.Y + p.ParticleRadius; if (velocity.Y < 0.0f) velocity.Y *= -p.Restitution; }
            if (position.Y >= p.Maximum.Y - p.ParticleRadius)
            { position.Y = p.Maximum.Y - p.ParticleRadius; if (velocity.Y > 0.0f) velocity.Y *= -p.Restitution; }
            particle.Position = Rotate(position, p.Cosine, p.Sine) + p.Center;
            particle.Velocity = Rotate(velocity, p.Cosine, p.Sine);
        }
        __global__ void Predict(Particle* particles, Float2* previous, Parameters p)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= p.Count) return;
            Particle particle = particles[i];
            previous[i] = particle.Position;
            particle.Velocity = particle.Velocity + p.TimeStep * p.Gravity;
            particle.Position = particle.Position + p.TimeStep * particle.Velocity;
            Constrain(particle, p); particles[i] = particle;
        }
        __device__ int CellX(Float2 position, Parameters p)
        { return max(0, min(p.GridWidth - 1, static_cast<int>(floorf((position.X - p.GridMinimum.X) / p.Radius)))); }
        __device__ int CellY(Float2 position, Parameters p)
        { return max(0, min(p.GridHeight - 1, static_cast<int>(floorf((position.Y - p.GridMinimum.Y) / p.Radius)))); }
        __global__ void BuildGrid(const Particle* particles, int* heads, int* next, Parameters p)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= p.Count) return;
            const int cell = CellY(particles[i].Position, p) * p.GridWidth + CellX(particles[i].Position, p);
            next[i] = atomicExch(heads + cell, i);
        }
        __device__ bool Kernel(Float2 delta, int i, int j, Parameters p, float& weight, Float2& gradient)
        {
            const float d2 = Dot(delta, delta), inverseRadius2 = 1.0f / (p.Radius * p.Radius);
            if (d2 >= p.Radius * p.Radius || i == j) return false;
            if (d2 < 1.0e-12f)
            {
                const float sign = i < j ? 1.0f : -1.0f;
                delta = ((i + j) & 1) ? Float2{sign * 0.001f * p.Radius, 0.0f} : Float2{0.0f, sign * 0.001f * p.Radius};
            }
            const float q = 1.0f - d2 * inverseRadius2;
            weight = q * q * q;
            gradient = (-6.0f * inverseRadius2 * q * q) * delta;
            return true;
        }
        __global__ void DensityLambda(Particle* particles, const int* heads, const int* next, float* lambdas, Parameters p)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= p.Count) return;
            const Float2 position = particles[i].Position;
            const int cx = CellX(position, p), cy = CellY(position, p);
            float density = 1.0f, gradientSquares = 0.0f; Float2 sum{0.0f, 0.0f};
            for (int y = max(0, cy - 1); y <= min(p.GridHeight - 1, cy + 1); ++y)
                for (int x = max(0, cx - 1); x <= min(p.GridWidth - 1, cx + 1); ++x)
                    for (int j = heads[y * p.GridWidth + x]; j >= 0; j = next[j])
                    {
                        float weight; Float2 gradient;
                        if (!Kernel(position - particles[j].Position, i, j, p, weight, gradient)) continue;
                        density += weight; gradient = p.InverseDensity * gradient;
                        sum = sum + gradient; gradientSquares += Dot(gradient, gradient);
                    }
            particles[i].Density = density;
            lambdas[i] = -fmaxf(density * p.InverseDensity - 1.0f, 0.0f) /
                (gradientSquares + Dot(sum, sum) + 0.01f / (p.Radius * p.Radius));
        }
        __global__ void Correct(const Particle* particles, Particle* target, const int* heads, const int* next,
                                const float* lambdas, Parameters p)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= p.Count) return;
            Particle particle = particles[i]; Float2 correction{0.0f, 0.0f};
            const int cx = CellX(particle.Position, p), cy = CellY(particle.Position, p);
            for (int y = max(0, cy - 1); y <= min(p.GridHeight - 1, cy + 1); ++y)
                for (int x = max(0, cx - 1); x <= min(p.GridWidth - 1, cx + 1); ++x)
                    for (int j = heads[y * p.GridWidth + x]; j >= 0; j = next[j])
                    {
                        float weight; Float2 gradient;
                        if (!Kernel(particle.Position - particles[j].Position, i, j, p, weight, gradient)) continue;
                        correction = correction + (p.Stiffness * (lambdas[i] + lambdas[j]) * p.InverseDensity) * gradient;
                    }
            const float length = sqrtf(Dot(correction, correction));
            particle.Position = particle.Position + fminf(1.0f, 0.2f * p.Radius / fmaxf(length, 1.0e-6f)) * correction;
            Constrain(particle, p); target[i] = particle;
        }
        __global__ void Reconstruct(Particle* particles, const Float2* previous, Parameters p)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < p.Count) particles[i].Velocity = (1.0f / p.TimeStep) * (particles[i].Position - previous[i]);
        }
        __global__ void Viscosity(const Particle* particles, Particle* target, const int* heads, const int* next, Parameters p)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= p.Count) return;
            Particle particle = particles[i]; Float2 exchange{0.0f, 0.0f};
            const int cx = CellX(particle.Position, p), cy = CellY(particle.Position, p);
            for (int y = max(0, cy - 1); y <= min(p.GridHeight - 1, cy + 1); ++y)
                for (int x = max(0, cx - 1); x <= min(p.GridWidth - 1, cx + 1); ++x)
                    for (int j = heads[y * p.GridWidth + x]; j >= 0; j = next[j])
                    {
                        float weight; Float2 gradient;
                        if (!Kernel(particle.Position - particles[j].Position, i, j, p, weight, gradient)) continue;
                        exchange = exchange + (p.Viscosity * weight / fmaxf(particle.Density, particles[j].Density)) *
                            (particles[j].Velocity - particle.Velocity);
                    }
            particle.Velocity = particle.Velocity + exchange;
            Constrain(particle, p); target[i] = particle;
        }

        class FluidSolver final : public CudaFluidSolver
        {
            Stream Queue;
            Buffer<Particle> Particles, Scratch;
            Buffer<Float2> Previous;
            Buffer<float> Lambdas;
            Buffer<int> Heads, Next;
            std::vector<FluidParticle2D> Readback;
        public:
            bool Step(const FluidSettings2D& settings, float restDensity, float deltaTime, int substeps,
                      std::vector<FluidParticle2D>& particles, std::string& error) override
            {
                try
                {
                    Check(cudaSetDevice(GetCudaBackendInfo().DeviceIndex)); Queue.Initialize();
                    const int count = static_cast<int>(particles.size());
                    if (!count) return true;
                    const float radius = std::max(settings.SmoothingRadius, 2.0f * settings.ParticleRadius);
                    if (radius <= 0.0f) throw std::runtime_error("Invalid CUDA fluid smoothing radius");
                    Vector2 minimum = Vector2::InfMax(), maximum = Vector2::InfMin();
                    for (float x : {settings.BoundsMinimum.X, settings.BoundsMaximum.X})
                        for (float y : {settings.BoundsMinimum.Y, settings.BoundsMaximum.Y})
                        {
                            const Vector2 corner = settings.ContainerTransform.TransformPoint({x, y});
                            minimum = minimum.Min(corner); maximum = maximum.Max(corner);
                        }
                    minimum -= radius;
                    const double width = std::ceil((maximum.X - minimum.X) / radius) + 2.0;
                    const double height = std::ceil((maximum.Y - minimum.Y) / radius) + 2.0;
                    if (!std::isfinite(width * height) || width * height > 4000000.0)
                        throw std::runtime_error("CUDA liquid grid exceeds capacity");
                    Parameters p{{settings.BoundsMinimum.X, settings.BoundsMinimum.Y}, {settings.BoundsMaximum.X, settings.BoundsMaximum.Y},
                        {settings.ContainerTransform.Position.X, settings.ContainerTransform.Position.Y}, {settings.Gravity.X, settings.Gravity.Y},
                        {minimum.X, minimum.Y}, std::cos(settings.ContainerTransform.Angle), std::sin(settings.ContainerTransform.Angle),
                        radius, settings.ParticleRadius, 1.0f / restDensity, std::clamp(settings.PressureStiffness, 0.0f, 1.0f),
                        1.0f - std::exp(-std::max(settings.Viscosity, 0.0f) * deltaTime * 60.0f), settings.BoundaryRestitution,
                        deltaTime, count, static_cast<int>(width), static_cast<int>(height)};
                    Particles.Reserve(count); Scratch.Reserve(count); Previous.Reserve(count);
                    Lambdas.Reserve(count); Next.Reserve(count); Heads.Reserve(p.GridWidth * p.GridHeight);
                    Readback.resize(count);
                    Check(cudaMemcpyAsync(Particles.Data, particles.data(), count * sizeof(Particle), cudaMemcpyHostToDevice, Queue.Handle));
                    auto rebuild = [&] {
                        Check(cudaMemsetAsync(Heads.Data, 0xff, p.GridWidth * p.GridHeight * sizeof(int), Queue.Handle));
                        BuildGrid<<<Blocks(count), 256, 0, Queue.Handle>>>(Particles.Data, Heads.Data, Next.Data, p);
                    };
                    for (int step = 0; step < substeps; ++step)
                    {
                        Predict<<<Blocks(count), 256, 0, Queue.Handle>>>(Particles.Data, Previous.Data, p);
                        for (int iteration = 0; iteration < std::max(settings.PressureIterations, 1); ++iteration)
                        {
                            rebuild();
                            DensityLambda<<<Blocks(count), 256, 0, Queue.Handle>>>(Particles.Data, Heads.Data, Next.Data, Lambdas.Data, p);
                            Correct<<<Blocks(count), 256, 0, Queue.Handle>>>(Particles.Data, Scratch.Data, Heads.Data, Next.Data, Lambdas.Data, p);
                            std::swap(Particles.Data, Scratch.Data);
                        }
                        Reconstruct<<<Blocks(count), 256, 0, Queue.Handle>>>(Particles.Data, Previous.Data, p);
                        rebuild();
                        DensityLambda<<<Blocks(count), 256, 0, Queue.Handle>>>(Particles.Data, Heads.Data, Next.Data, Lambdas.Data, p);
                        Viscosity<<<Blocks(count), 256, 0, Queue.Handle>>>(Particles.Data, Scratch.Data, Heads.Data, Next.Data, p);
                        std::swap(Particles.Data, Scratch.Data);
                    }
                    Check(cudaGetLastError());
                    Check(cudaMemcpyAsync(Readback.data(), Particles.Data, count * sizeof(Particle), cudaMemcpyDeviceToHost, Queue.Handle));
                    Check(cudaStreamSynchronize(Queue.Handle));
                    particles.swap(Readback); error.clear(); return true;
                }
                catch (const std::exception& exception)
                {
                    if (Queue.Handle) cudaStreamSynchronize(Queue.Handle);
                    error = exception.what(); return false;
                }
            }
        };
    }
    std::unique_ptr<CudaFluidSolver> CreateCudaFluidSolver() { return std::make_unique<FluidSolver>(); }
}
