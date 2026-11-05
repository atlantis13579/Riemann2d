#include "Continuum/SnowGpuSolver2D.h"
#include "Fluid/Cuda/CudaCommon.cuh"
#include "Fluid/FluidComputeBackend.h"

namespace Riemann2D
{
    namespace
    {
        using Continuum::Matrix2;
        using Cuda::Float2;
        struct Node { float X, Y, OldX, OldY, Mass; };

        __device__ Float2 Contact(Float2 position, Float2 velocity, const SnowCapsule2D& capsule, float padding)
        {
            if (!capsule.Enabled) return velocity;
            const Float2 closest{capsule.Center.X, Cuda::Clamp(position.Y, capsule.Center.Y - capsule.HalfLength, capsule.Center.Y + capsule.HalfLength)};
            const Float2 offset = position - closest;
            const float length = sqrtf(Cuda::Dot(offset, offset));
            if (length >= capsule.Radius + padding) return velocity;
            const Float2 normal = length > 1.e-6f ? (1 / length) * offset : Float2{1, 0};
            const Float2 relative = velocity - Float2{capsule.Velocity.X, capsule.Velocity.Y};
            const float incoming = Cuda::Dot(relative, normal);
            if (incoming >= 0) return velocity;
            const Float2 tangent = relative - incoming * normal;
            const float friction = fmaxf(0, 1 - capsule.Friction * (-incoming) / fmaxf(sqrtf(Cuda::Dot(tangent, tangent)), 1.e-6f));
            return Float2{capsule.Velocity.X, capsule.Velocity.Y} + friction * tangent;
        }

        __global__ void PrepareParticles(const SnowParticle2D* particles, Matrix2* stresses, int* heads, int* next,
                                         int count, int width, SnowSettings2D settings)
        {
            const int id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id >= count) return;
            const SnowParticle2D& p = particles[id];
            stresses[id] = Continuum::KirchhoffStress(p.Elastic, p.Affine, p.PlasticVolume, settings.Material);
            const int bx = static_cast<int>(floorf((p.Position.X - settings.BoundsMinimum.X) / settings.CellSize - 0.5f));
            const int by = static_cast<int>(floorf((p.Position.Y - settings.BoundsMinimum.Y) / settings.CellSize - 0.5f));
            next[id] = atomicExch(heads + by * width + bx, id);
        }

        __global__ void TransferGrid(const SnowParticle2D* particles, const Matrix2* stresses, const int* heads, const int* next,
                                    Node* nodes, int count, int width, int height, float dt,
                                    SnowSettings2D settings, SnowCapsule2D capsule)
        {
            const int id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id >= width * height) return;
            const int x = id % width, y = id / width;
            const float inverse = 1 / settings.CellSize;
            float mass = 0;
            Float2 momentum{0, 0}, force{0, 0};
            // Gather avoids floating-point atomics and mirrors the DX11/Metal path.
            for (int by = max(0, y - 2); by <= y; ++by)
                for (int bx = max(0, x - 2); bx <= x; ++bx)
                {
                    int visits = 0;
                    for (int index = heads[by * width + bx]; index >= 0 && visits++ < count; index = next[index])
                    {
                        const SnowParticle2D& p = particles[index];
                        const float fx = (p.Position.X - settings.BoundsMinimum.X) * inverse - bx;
                        const float fy = (p.Position.Y - settings.BoundsMinimum.Y) * inverse - by;
                        const int i = x - bx, j = y - by;
                        const float wx = Continuum::Weight(i, fx), wy = Continuum::Weight(j, fy);
                        const float weight = wx * wy;
                        const Float2 d{(i - fx) * settings.CellSize, (j - fy) * settings.CellSize};
                        const Float2 gradient = settings.Transfer == MpmTransfer2D::MLS ? (4 * inverse * inverse * weight) * d :
                            Float2{Continuum::WeightGradient(i, fx) * wy * inverse, Continuum::WeightGradient(j, fy) * wx * inverse};
                        const Matrix2 affine = settings.Transfer == MpmTransfer2D::MPM ? Matrix2{} : p.Affine;
                        const Matrix2 stress = stresses[index];
                        const float contribution = weight * p.Mass;
                        mass += contribution;
                        momentum = momentum + contribution * Float2{p.Velocity.X + affine.X(d.X, d.Y), p.Velocity.Y + affine.Y(d.X, d.Y)};
                        force = force - p.Volume * Float2{stress.X(gradient.X, gradient.Y), stress.Y(gradient.X, gradient.Y)};
                    }
                }
            Node node{};
            node.Mass = mass;
            if (mass > 0)
            {
                node.OldX = momentum.X / mass;
                node.OldY = momentum.Y / mass;
                Float2 velocity{node.OldX + dt * (force.X / mass + settings.Gravity.X), node.OldY + dt * (force.Y / mass + settings.Gravity.Y)};
                const Float2 position{settings.BoundsMinimum.X + x * settings.CellSize, settings.BoundsMinimum.Y + y * settings.CellSize};
                if (x < 2 && velocity.X < 0) velocity.X = 0;
                if (y < 2 && velocity.Y < 0) velocity.Y = 0;
                if (position.X > settings.BoundsMaximum.X - 2 * settings.CellSize && velocity.X > 0) velocity.X = 0;
                if (position.Y > settings.BoundsMaximum.Y - 2 * settings.CellSize && velocity.Y > 0) velocity.Y = 0;
                velocity = Contact(position, velocity, capsule, 0.25f * settings.CellSize);
                if (position.Y <= settings.FixedBedHeight) velocity = {0,0};
                node.X = velocity.X; node.Y = velocity.Y;
            }
            nodes[id] = node;
        }

        __global__ void TransferParticles(SnowParticle2D* particles, const Node* nodes, int count, int width, float dt,
                                         SnowSettings2D settings, SnowCapsule2D capsule)
        {
            const int id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id >= count) return;
            SnowParticle2D p = particles[id];
            if (p.Fixed > 0.5f) return;
            const float inverse = 1 / settings.CellSize;
            const float gridX = (p.Position.X - settings.BoundsMinimum.X) * inverse;
            const float gridY = (p.Position.Y - settings.BoundsMinimum.Y) * inverse;
            const int bx = static_cast<int>(floorf(gridX - 0.5f)), by = static_cast<int>(floorf(gridY - 0.5f));
            const float fx = gridX - bx, fy = gridY - by;
            Float2 pic{0,0}, change{0,0};
            Matrix2 gradient, affine;
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                {
                    const Node node = nodes[(by + j) * width + bx + i];
                    const float wx = Continuum::Weight(i, fx), wy = Continuum::Weight(j, fy);
                    const float weight = wx * wy;
                    const Float2 d{(i - fx) * settings.CellSize, (j - fy) * settings.CellSize};
                    const Float2 dw = settings.Transfer == MpmTransfer2D::MLS ? (4 * inverse * inverse * weight) * d :
                        Float2{Continuum::WeightGradient(i, fx) * wy * inverse, Continuum::WeightGradient(j, fy) * wx * inverse};
                    pic = pic + weight * Float2{node.X, node.Y};
                    change = change + weight * Float2{node.X - node.OldX, node.Y - node.OldY};
                    gradient = gradient + Matrix2(node.X * dw.X, node.X * dw.Y, node.Y * dw.X, node.Y * dw.Y);
                    affine = affine + Matrix2(node.X * weight * d.X, node.X * weight * d.Y, node.Y * weight * d.X, node.Y * weight * d.Y) * (4 * inverse * inverse);
                }
            Float2 velocity = pic;
            if (settings.Transfer == MpmTransfer2D::MPM)
                velocity = (1 - settings.FlipRatio) * pic + settings.FlipRatio * (Float2{p.Velocity.X, p.Velocity.Y} + change);
            p.Affine = affine;
            p.Elastic = Continuum::ProjectPlasticity((Matrix2::Identity() + gradient * dt) * p.Elastic, p.PlasticVolume, settings.Material);
            Float2 position{p.Position.X + dt * pic.X, p.Position.Y + dt * pic.Y};
            const Float2 low{settings.BoundsMinimum.X + 1.5f * settings.CellSize, settings.BoundsMinimum.Y + 1.5f * settings.CellSize};
            const Float2 high{settings.BoundsMaximum.X - 1.5f * settings.CellSize, settings.BoundsMaximum.Y - 1.5f * settings.CellSize};
            if (position.X < low.X || position.X > high.X) velocity.X = 0;
            if (position.Y < low.Y || position.Y > high.Y) velocity.Y = 0;
            position = {Cuda::Clamp(position.X, low.X, high.X), Cuda::Clamp(position.Y, low.Y, high.Y)};
            velocity = Contact(position, velocity, capsule, 0);
            if (capsule.Enabled)
            {
                const Float2 closest{capsule.Center.X, Cuda::Clamp(position.Y, capsule.Center.Y - capsule.HalfLength, capsule.Center.Y + capsule.HalfLength)};
                const Float2 offset = position - closest;
                const float length = sqrtf(Cuda::Dot(offset, offset));
                if (length < capsule.Radius)
                    position = closest + capsule.Radius * (length > 1.e-6f ? (1 / length) * offset : Float2{1,0});
            }
            p.Position.X = Cuda::Clamp(position.X, low.X, high.X);
            p.Position.Y = Cuda::Clamp(position.Y, low.Y, high.Y);
            p.Velocity.X = velocity.X; p.Velocity.Y = velocity.Y;
            particles[id] = p;
        }

        class CudaSnowSolver final : public SnowGpuSolver2D
        {
            Cuda::Buffer<SnowParticle2D> Particles;
            Cuda::Buffer<Matrix2> Stresses;
            Cuda::Buffer<Node> Nodes;
            Cuda::Buffer<int> Heads, Next;
            Cuda::Stream Queue;
            std::vector<SnowParticle2D> Readback;
        public:
            bool Step(const SnowSettings2D& settings, const SnowCapsule2D& capsule, float dt,
                      std::vector<SnowParticle2D>& particles, std::string& error) override
            {
                try
                {
                    Cuda::Check(cudaSetDevice(GetCudaBackendInfo().DeviceIndex));
                    Queue.Initialize();
                    if (particles.empty()) return true;
                    const int count = static_cast<int>(particles.size());
                    const int width = static_cast<int>(std::ceil((settings.BoundsMaximum.X - settings.BoundsMinimum.X) / settings.CellSize)) + 1;
                    const int height = static_cast<int>(std::ceil((settings.BoundsMaximum.Y - settings.BoundsMinimum.Y) / settings.CellSize)) + 1;
                    const int cells = width * height;
                    Particles.Reserve(count); Stresses.Reserve(count); Next.Reserve(count);
                    Nodes.Reserve(cells); Heads.Reserve(cells); Readback.resize(count);
                    Cuda::Check(cudaMemcpyAsync(Particles.Data, particles.data(), count * sizeof(SnowParticle2D), cudaMemcpyHostToDevice, Queue.Handle));
                    Cuda::Check(cudaMemsetAsync(Heads.Data, 0xff, cells * sizeof(int), Queue.Handle));
                    PrepareParticles<<<Cuda::Blocks(count),256,0,Queue.Handle>>>(Particles.Data, Stresses.Data, Heads.Data, Next.Data, count, width, settings);
                    Cuda::Check(cudaGetLastError());
                    TransferGrid<<<Cuda::Blocks(cells),256,0,Queue.Handle>>>(Particles.Data, Stresses.Data, Heads.Data, Next.Data, Nodes.Data, count, width, height, dt, settings, capsule);
                    Cuda::Check(cudaGetLastError());
                    TransferParticles<<<Cuda::Blocks(count),256,0,Queue.Handle>>>(Particles.Data, Nodes.Data, count, width, dt, settings, capsule);
                    Cuda::Check(cudaGetLastError());
                    Cuda::Check(cudaMemcpyAsync(Readback.data(), Particles.Data, count * sizeof(SnowParticle2D), cudaMemcpyDeviceToHost, Queue.Handle));
                    Cuda::Check(cudaStreamSynchronize(Queue.Handle));
                    for (const auto& p : Readback)
                        if (!std::isfinite(p.Position.X + p.Position.Y + p.Velocity.X + p.Velocity.Y + p.Elastic.Determinant() + p.PlasticVolume))
                            throw std::runtime_error("CUDA MPM produced a non-finite state");
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

    std::unique_ptr<SnowGpuSolver2D> CreateCudaSnowSolver2D() { return std::make_unique<CudaSnowSolver>(); }
}
