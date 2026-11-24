#include "CudaSolvers.h"
#include "CudaCommon.cuh"
#include "Fluid/AerodynamicsSimulation2D.h"

#include <algorithm>
#include <cmath>

namespace Riemann2D
{
    namespace
    {
        using namespace Cuda;
        struct AirParameters
        {
            int Width, Height;
            float InverseX, InverseY, TimeStep, Diffusion, Vorticity, Decay;
            int MacCormack;
        };
        struct Level
        {
            int Width, Height;
            float WeightX, WeightY;
            float *Pressure, *RHS, *Residual, *InverseDiagonal, *Mean;
            int* Solid;
        };
        struct Sample
        {
            int A, B, C, D;
            float X, Y;
            __device__ Sample(int w, int h, float x, float y)
            {
                x = Cuda::Clamp(x, 0.0f, static_cast<float>(w - 1));
                y = Cuda::Clamp(y, 0.0f, static_cast<float>(h - 1));
                const int ix = static_cast<int>(x), iy = static_cast<int>(y);
                const int nx = min(ix + 1, w - 1), ny = min(iy + 1, h - 1);
                A = iy * w + ix; B = iy * w + nx; C = ny * w + ix; D = ny * w + nx;
                X = x - ix; Y = y - iy;
            }
            __device__ float Read(const float* f) const
            { return (f[A] + X * (f[B] - f[A])) * (1.0f - Y) + (f[C] + X * (f[D] - f[C])) * Y; }
            __device__ float Limit(float value, const float* f) const
            { return Cuda::Clamp(value, fminf(fminf(f[A], f[B]), fminf(f[C], f[D])), fmaxf(fmaxf(f[A], f[B]), fmaxf(f[C], f[D]))); }
        };
        __device__ Float2 Velocity(const float* u, const float* v, AirParameters p, float x, float y)
        { return {Sample(p.Width + 1, p.Height, x, y - 0.5f).Read(u), Sample(p.Width, p.Height + 1, x - 0.5f, y).Read(v)}; }
        __device__ Float2 Center(const float* u, const float* v, int x, int y, int w)
        { return {0.5f * (u[y * (w + 1) + x] + u[y * (w + 1) + x + 1]), 0.5f * (v[y * w + x] + v[(y + 1) * w + x])}; }
        __global__ void AdvectFaces(const float* u, const float* v, float* result, int component, const AirParameters* parameters)
        {
            const AirParameters p = *parameters;
            const int w = p.Width + (component == 0), h = p.Height + (component == 1);
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= w * h) return;
            const int x = i % w, y = i / w;
            if ((component == 0 && (x == 0 || x == w - 1)) || (component == 1 && (y == 0 || y == h - 1)))
            { result[i] = 0.0f; return; }
            const float gx = x + (component == 1 ? 0.5f : 0.0f), gy = y + (component == 0 ? 0.5f : 0.0f);
            const Float2 velocity = Velocity(u, v, p, gx, gy);
            const float dtX = p.TimeStep * p.InverseX, dtY = p.TimeStep * p.InverseY;
            const Float2 midpoint = Velocity(u, v, p, gx - 0.5f * dtX * velocity.X, gy - 0.5f * dtY * velocity.Y);
            const float* source = component == 0 ? u : v;
            float value = Sample(w, h, x - dtX * midpoint.X, y - dtY * midpoint.Y).Read(source);
            const float ax = p.Diffusion * p.TimeStep * p.InverseX * p.InverseX;
            const float ay = p.Diffusion * p.TimeStep * p.InverseY * p.InverseY;
            const float scale = fminf(1.0f, 0.49f / fmaxf(ax + ay, 1.0e-6f));
            if (x > 0 && x + 1 < w && y > 0 && y + 1 < h)
                value += scale * (ax * (source[i - 1] + source[i + 1] - 2.0f * source[i]) +
                                 ay * (source[i - w] + source[i + w] - 2.0f * source[i]));
            result[i] = value;
        }
        __global__ void ComputeCenters(const float* u, const float* v, Float2* centers, int w, int h)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < w * h) centers[i] = Center(u, v, i % w, i / w, w);
        }
        __global__ void ComputeCurl(const Float2* velocity, float* curl, const AirParameters* parameters)
        {
            const AirParameters p = *parameters;
            const int i = blockIdx.x * blockDim.x + threadIdx.x, w = p.Width, h = p.Height;
            if (i >= w * h) return;
            const int x = i % w, y = i / w;
            curl[i] = x > 0 && x + 1 < w && y > 0 && y + 1 < h ?
                0.5f * ((velocity[i + 1].Y - velocity[i - 1].Y) * p.InverseX -
                        (velocity[i + w].X - velocity[i - w].X) * p.InverseY) : 0.0f;
        }
        __global__ void ComputeForce(const float* curl, Float2* force, const AirParameters* parameters)
        {
            const AirParameters p = *parameters;
            const int i = blockIdx.x * blockDim.x + threadIdx.x, w = p.Width, h = p.Height;
            if (i >= w * h) return;
            const int x = i % w, y = i / w;
            Float2 value{0.0f, 0.0f};
            if (p.Vorticity > 0.0f && x > 0 && x + 1 < w && y > 0 && y + 1 < h)
            {
                const Float2 gradient{(fabsf(curl[i + 1]) - fabsf(curl[i - 1])) * p.InverseX,
                                      (fabsf(curl[i + w]) - fabsf(curl[i - w])) * p.InverseY};
                const float scale = p.Vorticity * fminf(1.0f / p.InverseX, 1.0f / p.InverseY) * curl[i] /
                    fmaxf(sqrtf(Dot(gradient, gradient)), 1.0e-5f);
                value = scale * Float2{gradient.Y, -gradient.X};
            }
            force[i] = value;
        }
        __global__ void ApplyForce(float* u, float* v, const Float2* force, const AirParameters* parameters)
        {
            const AirParameters p = *parameters;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < (p.Width + 1) * p.Height)
            {
                const int x = i % (p.Width + 1), y = i / (p.Width + 1);
                if (x > 0 && x < p.Width) u[i] += 0.5f * p.TimeStep * (force[y * p.Width + x - 1].X + force[y * p.Width + x].X);
            }
            if (i < p.Width * (p.Height + 1))
            {
                const int x = i % p.Width, y = i / p.Width;
                if (y > 0 && y < p.Height) v[i] += 0.5f * p.TimeStep * (force[(y - 1) * p.Width + x].Y + force[y * p.Width + x].Y);
            }
        }
        __global__ void PreparePressure(const float* u, const float* v, Level level, const AirParameters* parameters)
        {
            const AirParameters p = *parameters;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= p.Width * p.Height) return;
            const int x = i % p.Width, y = i / p.Width;
            if (level.Solid[i] >= 0) { level.Pressure[i] = level.RHS[i] = 0; return; }
            level.Pressure[i] = 0.0f;
            level.RHS[i] = -(u[y * (p.Width + 1) + x + 1] - u[y * (p.Width + 1) + x]) * p.InverseX -
                           (v[(y + 1) * p.Width + x] - v[y * p.Width + x]) * p.InverseY;
        }
        __device__ float NeighborSum(const float* pressure, Level l, int x, int y, int i)
        {
            return ((x > 0 && l.Solid[i - 1] < 0 ? pressure[i - 1] : 0.0f) + (x + 1 < l.Width && l.Solid[i + 1] < 0 ? pressure[i + 1] : 0.0f)) * l.WeightX +
                   ((y > 0 && l.Solid[i - l.Width] < 0 ? pressure[i - l.Width] : 0.0f) + (y + 1 < l.Height && l.Solid[i + l.Width] < 0 ? pressure[i + l.Width] : 0.0f)) * l.WeightY;
        }
        __global__ void Smooth(Level level, int parity)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= level.Width * level.Height) return;
            const int x = i % level.Width, y = i / level.Width;
            if (((x + y) & 1) != parity) return;
            const float sum = NeighborSum(level.Pressure, level, x, y, i);
            level.Pressure[i] += 1.15f * ((sum + level.RHS[i]) * level.InverseDiagonal[i] - level.Pressure[i]);
        }
        __global__ void SmoothCoarsest(Level level)
        {
            __shared__ float pressure[16];
            const int i = threadIdx.x, count = level.Width * level.Height;
            if (i < count) pressure[i] = level.Pressure[i];
            __syncthreads();
            for (int iteration = 0; iteration < 40; ++iteration)
                for (int parity = 0; parity < 2; ++parity)
                {
                    if (i < count)
                    {
                        const int x = i % level.Width, y = i / level.Width;
                        if (((x + y) & 1) == parity)
                            pressure[i] += 1.15f * ((NeighborSum(pressure, level, x, y, i) + level.RHS[i]) * level.InverseDiagonal[i] - pressure[i]);
                    }
                    __syncthreads();
                }
            if (i < count) level.Pressure[i] = pressure[i];
        }
        __global__ void Residual(Level level)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= level.Width * level.Height) return;
            if (level.InverseDiagonal[i] == 0) { level.Residual[i] = 0; return; }
            level.Residual[i] = level.RHS[i] - level.Pressure[i] / level.InverseDiagonal[i] +
                NeighborSum(level.Pressure, level, i % level.Width, i / level.Width, i);
        }
        __global__ void Restrict(Level fine, Level coarse)
        {
            __shared__ float sums[256];
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            float value = 0.0f;
            if (i < coarse.Width * coarse.Height)
            {
                const int x = i % coarse.Width, y = i / coarse.Width;
                int count = 0;
                for (int dy = 0; dy < 2 && 2 * y + dy < fine.Height; ++dy)
                    for (int dx = 0; dx < 2 && 2 * x + dx < fine.Width; ++dx)
                    { value += fine.Residual[(2 * y + dy) * fine.Width + 2 * x + dx]; ++count; }
                value /= count;
                coarse.RHS[i] = value; coarse.Pressure[i] = 0.0f;
            }
            sums[threadIdx.x] = value; __syncthreads();
            for (int stride = 128; stride > 0; stride /= 2)
            {
                if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0) atomicAdd(coarse.Mean, sums[0]);
        }
        __global__ void RemoveMean(Level level)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x, count = level.Width * level.Height;
            if (i < count) level.RHS[i] -= *level.Mean / count;
        }
        __global__ void Prolong(Level fine, Level coarse)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= fine.Width * fine.Height || fine.Solid[i] >= 0) return;
            fine.Pressure[i] += Sample(coarse.Width, coarse.Height,
                (i % fine.Width + 0.5f) * coarse.Width / fine.Width - 0.5f,
                (i / fine.Width + 0.5f) * coarse.Height / fine.Height - 0.5f).Read(coarse.Pressure);
        }
        __global__ void ApplyPressure(float* u, float* v, const float* pressure, const AirParameters* parameters)
        {
            const AirParameters p = *parameters;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < (p.Width + 1) * p.Height)
            {
                const int x = i % (p.Width + 1), y = i / (p.Width + 1);
                if (x > 0 && x < p.Width) u[i] -= (pressure[y * p.Width + x] - pressure[y * p.Width + x - 1]) * p.InverseX;
                else u[i] = 0.0f;
            }
            if (i < p.Width * (p.Height + 1))
            {
                const int x = i % p.Width, y = i / p.Width;
                if (y > 0 && y < p.Height) v[i] -= (pressure[y * p.Width + x] - pressure[(y - 1) * p.Width + x]) * p.InverseY;
                else v[i] = 0.0f;
            }
        }
        __device__ Float2 ClipTrace(Float2 start, Float2 end, const int* solid, int w, int h)
        {
            Float2 delta = end - start, previous = start;
            int steps = max(1, (int)ceilf(fmaxf(fabsf(delta.X),fabsf(delta.Y))*3));
            for (int i = 1; i <= steps; ++i)
            {
                Float2 p = start + ((float)i/steps)*delta;
                int x = (int)floorf(p.X+.5f), y = (int)floorf(p.Y+.5f);
                if (x >= 0 && x < w && y >= 0 && y < h && solid[y*w+x] >= 0) return previous;
                previous = p;
            }
            return end;
        }
        __global__ void ForwardDensity(const float* density, const float* u, const float* v, float* forward,
                                       Float2* centers, const int* solid, const AirParameters* parameters)
        {
            const AirParameters p = *parameters;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= p.Width * p.Height) return;
            const int x = i % p.Width, y = i / p.Width;
            const Float2 velocity = Center(u, v, x, y, p.Width); centers[i] = velocity;
            const auto donor = ClipTrace({(float)x,(float)y}, {x-p.TimeStep*p.InverseX*velocity.X,y-p.TimeStep*p.InverseY*velocity.Y}, solid, p.Width,p.Height);
            forward[i] = solid[i] >= 0 ? 0 : Sample(p.Width,p.Height,donor.X,donor.Y).Read(density);
        }
        __global__ void CorrectDensity(const float* density, const float* forward, const Float2* centers,
                                       float* target, const int* solid, const AirParameters* parameters)
        {
            const AirParameters p = *parameters;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= p.Width * p.Height) return;
            const int x = i % p.Width, y = i / p.Width;
            const float dx = p.TimeStep * p.InverseX * centers[i].X, dy = p.TimeStep * p.InverseY * centers[i].Y;
            float value = forward[i];
            if (p.MacCormack)
            {
                const auto ahead = ClipTrace({(float)x,(float)y},{x+dx,y+dy},solid,p.Width,p.Height);
                const auto back = ClipTrace({(float)x,(float)y},{x-dx,y-dy},solid,p.Width,p.Height);
                const float reverse = Sample(p.Width,p.Height,ahead.X,ahead.Y).Read(forward);
                value = Sample(p.Width,p.Height,back.X,back.Y).Limit(value + 0.5f * (density[i] - reverse), density);
            }
            target[i] = solid[i] >= 0 ? 0 : p.Decay * fmaxf(0.0f, value);
        }

        __global__ void CoarsenMask(Level fine, Level coarse)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= coarse.Width * coarse.Height) return;
            int x = 2 * (i % coarse.Width), y = 2 * (i / coarse.Width), solid = 0;
            for (int dy = 0; dy < 2 && y + dy < fine.Height; ++dy)
                for (int dx = 0; dx < 2 && x + dx < fine.Width; ++dx)
                    if (fine.Solid[(y + dy) * fine.Width + x + dx] < 0) solid = -1;
            coarse.Solid[i] = solid;
        }
        __global__ void MaskDiagonal(Level l)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= l.Width * l.Height) return;
            int x = i % l.Width, y = i / l.Width;
            float diagonal = l.Solid[i] >= 0 ? 0 :
                ((x > 0 && l.Solid[i-1] < 0) + (x+1 < l.Width && l.Solid[i+1] < 0)) * l.WeightX +
                ((y > 0 && l.Solid[i-l.Width] < 0) + (y+1 < l.Height && l.Solid[i+l.Width] < 0)) * l.WeightY;
            l.InverseDiagonal[i] = diagonal > 0 ? 1.0f / diagonal : 0;
        }
        __global__ void DisplaceDensity(const float* density, float* result, const int* solid,
                                        const int* head, const int* next, int count)
        {
            int i = blockIdx.x*blockDim.x+threadIdx.x;
            if (i >= count) return;
            float value = solid[i] < 0 ? density[i] : 0;
            for (int source = head[i]; source >= 0; source = next[source]) value += density[source];
            result[i] = value;
        }
        __global__ void EnforceSolid(float* u, float* v, const int* bu, const int* bv,
                                    const float* vu, const float* vv, const AirParameters* parameters)
        {
            const auto p = *parameters;
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < (p.Width+1)*p.Height && bu[i] >= 0) u[i] = vu[i];
            if (i < p.Width*(p.Height+1) && bv[i] >= 0) v[i] = vv[i];
        }
        struct LevelBuffers
        {
            Buffer<float> Pressure, RHS, Residual, InverseDiagonal, Mean;
            Buffer<int> Solid;
        };
        struct Graph
        {
            cudaGraphExec_t Instance = nullptr;
            ~Graph() { Reset(); }
            void Reset() { if (Instance) cudaGraphExecDestroy(Instance); Instance = nullptr; }
        };
        class AirSolver final : public CudaAirSolver
        {
            Stream Queue;
            Buffer<float> U, V, UScratch, VScratch, Density, Forward, DensityScratch, Curl;
            Buffer<Float2> Centers, Force;
            Buffer<AirParameters> Parameters;
            Buffer<int> BlockU, BlockV, DisplacementHead, DisplacementNext;
            Buffer<float> BoundaryU, BoundaryV;
            bool HasBoundary = false;
            std::vector<LevelBuffers> Storage;
            std::vector<Level> Levels;
            std::vector<float> ReadU, ReadV, ReadDensity;
            Graph StepGraph, ProjectionGraph;
            int Width = 0, Height = 0, Cycles = 0;
            Vector2 CellSize;

            void Cycle(std::size_t index)
            {
                Level level = Levels[index];
                if (index + 1 == Levels.size()) { SmoothCoarsest<<<1, 32, 0, Queue.Handle>>>(level); return; }
                for (int iteration = 0; iteration < 2; ++iteration)
                    for (int parity = 0; parity < 2; ++parity) Smooth<<<Blocks(level.Width * level.Height), 256, 0, Queue.Handle>>>(level, parity);
                Residual<<<Blocks(level.Width * level.Height), 256, 0, Queue.Handle>>>(level);
                Level coarse = Levels[index + 1];
                Check(cudaMemsetAsync(coarse.Mean, 0, sizeof(float), Queue.Handle));
                Restrict<<<Blocks(coarse.Width * coarse.Height), 256, 0, Queue.Handle>>>(level, coarse);
                if (!HasBoundary) RemoveMean<<<Blocks(coarse.Width * coarse.Height), 256, 0, Queue.Handle>>>(coarse);
                Cycle(index + 1);
                Prolong<<<Blocks(level.Width * level.Height), 256, 0, Queue.Handle>>>(level, coarse);
                for (int iteration = 0; iteration < 2; ++iteration)
                    for (int parity = 0; parity < 2; ++parity) Smooth<<<Blocks(level.Width * level.Height), 256, 0, Queue.Handle>>>(level, parity);
            }
            void SynchronizeBoundary()
            {
                if (!HasBoundary) return;
                const int n = Width*Height;
                DisplaceDensity<<<Blocks(n),256,0,Queue.Handle>>>(Density.Data,DensityScratch.Data,
                    Storage[0].Solid.Data,DisplacementHead.Data,DisplacementNext.Data,n);
                Check(cudaMemcpyAsync(Density.Data,DensityScratch.Data,n*sizeof(float),cudaMemcpyDeviceToDevice,Queue.Handle));
                EnforceSolid<<<Blocks(std::max((Width+1)*Height,Width*(Height+1))),256,0,Queue.Handle>>>(
                    U.Data,V.Data,BlockU.Data,BlockV.Data,BoundaryU.Data,BoundaryV.Data,Parameters.Data);
            }
            void Capture(Graph& graph, bool projectOnly)
            {
                Check(cudaStreamBeginCapture(Queue.Handle, cudaStreamCaptureModeThreadLocal));
                cudaGraph_t captured = nullptr;
                try
                {
                    const int n = Width * Height, nu = (Width + 1) * Height, nv = Width * (Height + 1);
                    SynchronizeBoundary();
                    if (!projectOnly)
                    {
                        AdvectFaces<<<Blocks(nu), 256, 0, Queue.Handle>>>(U.Data, V.Data, UScratch.Data, 0, Parameters.Data);
                        AdvectFaces<<<Blocks(nv), 256, 0, Queue.Handle>>>(U.Data, V.Data, VScratch.Data, 1, Parameters.Data);
                        Check(cudaMemcpyAsync(U.Data, UScratch.Data, nu * sizeof(float), cudaMemcpyDeviceToDevice, Queue.Handle));
                        Check(cudaMemcpyAsync(V.Data, VScratch.Data, nv * sizeof(float), cudaMemcpyDeviceToDevice, Queue.Handle));
                        ComputeCenters<<<Blocks(n), 256, 0, Queue.Handle>>>(U.Data, V.Data, Centers.Data, Width, Height);
                        ComputeCurl<<<Blocks(n), 256, 0, Queue.Handle>>>(Centers.Data, Curl.Data, Parameters.Data);
                        ComputeForce<<<Blocks(n), 256, 0, Queue.Handle>>>(Curl.Data, Force.Data, Parameters.Data);
                        ApplyForce<<<Blocks(std::max(nu, nv)), 256, 0, Queue.Handle>>>(U.Data, V.Data, Force.Data, Parameters.Data);
                    }
                    if (HasBoundary)
                        EnforceSolid<<<Blocks(std::max(nu,nv)),256,0,Queue.Handle>>>(U.Data,V.Data,BlockU.Data,BlockV.Data,BoundaryU.Data,BoundaryV.Data,Parameters.Data);
                    for (std::size_t l = 0; l < Levels.size(); ++l)
                    {
                        if (l) CoarsenMask<<<Blocks(Levels[l].Width*Levels[l].Height),256,0,Queue.Handle>>>(Levels[l-1],Levels[l]);
                        MaskDiagonal<<<Blocks(Levels[l].Width*Levels[l].Height),256,0,Queue.Handle>>>(Levels[l]);
                    }
                    PreparePressure<<<Blocks(n), 256, 0, Queue.Handle>>>(U.Data, V.Data, Levels[0], Parameters.Data);
                    for (int iteration = 0; iteration < Cycles; ++iteration) Cycle(0);
                    ApplyPressure<<<Blocks(std::max(nu, nv)), 256, 0, Queue.Handle>>>(U.Data, V.Data, Levels[0].Pressure, Parameters.Data);
                    if (HasBoundary)
                        EnforceSolid<<<Blocks(std::max(nu,nv)),256,0,Queue.Handle>>>(U.Data,V.Data,BlockU.Data,BlockV.Data,BoundaryU.Data,BoundaryV.Data,Parameters.Data);
                    if (!projectOnly)
                    {
                        ForwardDensity<<<Blocks(n), 256, 0, Queue.Handle>>>(Density.Data, U.Data, V.Data, Forward.Data, Centers.Data, Storage[0].Solid.Data, Parameters.Data);
                        CorrectDensity<<<Blocks(n), 256, 0, Queue.Handle>>>(Density.Data, Forward.Data, Centers.Data, DensityScratch.Data, Storage[0].Solid.Data, Parameters.Data);
                        Check(cudaMemcpyAsync(Density.Data, DensityScratch.Data, n * sizeof(float), cudaMemcpyDeviceToDevice, Queue.Handle));
                    }
                    Check(cudaGetLastError());
                    Check(cudaStreamEndCapture(Queue.Handle, &captured));
                    Check(cudaGraphInstantiate(&graph.Instance, captured, nullptr, nullptr, 0));
                    Check(cudaGraphDestroy(captured));
                }
                catch (...)
                {
                    cudaStreamCaptureStatus status;
                    if (cudaStreamIsCapturing(Queue.Handle, &status) == cudaSuccess && status != cudaStreamCaptureStatusNone)
                        cudaStreamEndCapture(Queue.Handle, &captured);
                    if (captured) cudaGraphDestroy(captured);
                    throw;
                }
            }
            void Resize(int width, int height, Vector2 cellSize)
            {
                StepGraph.Reset(); ProjectionGraph.Reset(); Levels.clear(); Storage.clear();
                Width = width; Height = height; CellSize = cellSize;
                const int n = width * height, nu = (width + 1) * height, nv = width * (height + 1);
                U.Reserve(nu); V.Reserve(nv); UScratch.Reserve(nu); VScratch.Reserve(nv);
                Density.Reserve(n); Forward.Reserve(n); DensityScratch.Reserve(n); Curl.Reserve(n);
                Centers.Reserve(n); Force.Reserve(n); Parameters.Reserve(1);
                DisplacementHead.Reserve(n); DisplacementNext.Reserve(n);
                BlockU.Reserve(nu); BlockV.Reserve(nv); BoundaryU.Reserve(nu); BoundaryV.Reserve(nv);
                ReadU.resize(nu); ReadV.resize(nv); ReadDensity.resize(n);
                const Vector2 extent{width * cellSize.X, height * cellSize.Y};
                for (;;)
                {
                    Storage.emplace_back(); auto& storage = Storage.back();
                    const int count = width * height;
                    storage.Pressure.Reserve(count); storage.RHS.Reserve(count); storage.Residual.Reserve(count);
                    storage.Solid.Reserve(count);
                    storage.InverseDiagonal.Reserve(count); storage.Mean.Reserve(1);
                    const float inverseX = width / extent.X, inverseY = height / extent.Y;
                    const float wx = inverseX * inverseX, wy = inverseY * inverseY;
                    std::vector<float> inverse(count);
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x)
                            inverse[y * width + x] = 1.0f / (((x > 0) + (x + 1 < width)) * wx + ((y > 0) + (y + 1 < height)) * wy);
                    Check(cudaMemcpy(storage.InverseDiagonal.Data, inverse.data(), count * sizeof(float), cudaMemcpyHostToDevice));
                    Levels.push_back({width, height, wx, wy, storage.Pressure.Data, storage.RHS.Data,
                        storage.Residual.Data, storage.InverseDiagonal.Data, storage.Mean.Data, storage.Solid.Data});
                    if (width <= 4 && height <= 4) break;
                    width = (width + 1) / 2; height = (height + 1) / 2;
                }
            }
        public:
            bool Step(const AerodynamicsSettings2D& settings, const Vector2& cellSize,
                      int width, int height, float deltaTime, bool projectOnly, const AirBoundaryData* boundary, bool synchronizeOnly, std::vector<float>& density,
                      std::vector<float>& u, std::vector<float>& v, std::vector<float>& pressure, std::string& error) override
            {
                try
                {
                    Check(cudaSetDevice(GetCudaBackendInfo().DeviceIndex)); Queue.Initialize();
                    const int cycles = std::max(boundary ? 24 : 1, (settings.PressureIterations + 7) / 8);
                    if (width != Width || height != Height || cellSize.X != CellSize.X || cellSize.Y != CellSize.Y)
                        Resize(width, height, cellSize);
                    if (HasBoundary != (boundary != nullptr)) { StepGraph.Reset(); ProjectionGraph.Reset(); HasBoundary = boundary != nullptr; }
                    if (Cycles != cycles) { StepGraph.Reset(); ProjectionGraph.Reset(); Cycles = cycles; }
                    const AirParameters p{width, height, 1.0f / cellSize.X, 1.0f / cellSize.Y, deltaTime,
                        std::max(settings.VelocityDiffusion, 0.0f), settings.VorticityStrength,
                        std::exp(-std::max(settings.DensityDissipation, 0.0f) * deltaTime), settings.UseMacCormack ? 1 : 0};
                    Check(cudaMemcpyAsync(Parameters.Data, &p, sizeof(p), cudaMemcpyHostToDevice, Queue.Handle));
                    Check(cudaMemcpyAsync(U.Data, u.data(), u.size() * sizeof(float), cudaMemcpyHostToDevice, Queue.Handle));
                    Check(cudaMemcpyAsync(V.Data, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice, Queue.Handle));
                    Check(cudaMemcpyAsync(Density.Data, density.data(), density.size() * sizeof(float), cudaMemcpyHostToDevice, Queue.Handle));
                    if (boundary)
                    {
                        Check(cudaMemcpyAsync(DisplacementHead.Data,boundary->DisplacementHead.data(),width*height*sizeof(int),cudaMemcpyHostToDevice,Queue.Handle));
                        Check(cudaMemcpyAsync(DisplacementNext.Data,boundary->DisplacementNext.data(),width*height*sizeof(int),cudaMemcpyHostToDevice,Queue.Handle));
                        Check(cudaMemcpyAsync(Storage[0].Solid.Data,boundary->Solid.data(),width*height*sizeof(int),cudaMemcpyHostToDevice,Queue.Handle));
                        Check(cudaMemcpyAsync(BlockU.Data,boundary->BlockU.data(),u.size()*sizeof(int),cudaMemcpyHostToDevice,Queue.Handle));
                        Check(cudaMemcpyAsync(BlockV.Data,boundary->BlockV.data(),v.size()*sizeof(int),cudaMemcpyHostToDevice,Queue.Handle));
                        Check(cudaMemcpyAsync(BoundaryU.Data,boundary->VelocityU.data(),u.size()*sizeof(float),cudaMemcpyHostToDevice,Queue.Handle));
                        Check(cudaMemcpyAsync(BoundaryV.Data,boundary->VelocityV.data(),v.size()*sizeof(float),cudaMemcpyHostToDevice,Queue.Handle));
                    }
                    else Check(cudaMemsetAsync(Storage[0].Solid.Data,255,width*height*sizeof(int),Queue.Handle));
                    if (synchronizeOnly) SynchronizeBoundary();
                    else
                    {
                        Graph& graph = projectOnly ? ProjectionGraph : StepGraph;
                        if (!graph.Instance) Capture(graph, projectOnly);
                        Check(cudaGraphLaunch(graph.Instance, Queue.Handle));
                    }
                    pressure.resize(static_cast<std::size_t>(width) * height);
                    if (!synchronizeOnly) Check(cudaMemcpyAsync(pressure.data(), Levels[0].Pressure, pressure.size() * sizeof(float), cudaMemcpyDeviceToHost, Queue.Handle));
                    Check(cudaMemcpyAsync(ReadU.data(), U.Data, u.size() * sizeof(float), cudaMemcpyDeviceToHost, Queue.Handle));
                    Check(cudaMemcpyAsync(ReadV.data(), V.Data, v.size() * sizeof(float), cudaMemcpyDeviceToHost, Queue.Handle));
                    Check(cudaMemcpyAsync(ReadDensity.data(), Density.Data, density.size() * sizeof(float), cudaMemcpyDeviceToHost, Queue.Handle));
                    Check(cudaStreamSynchronize(Queue.Handle));
                    u.swap(ReadU); v.swap(ReadV); density.swap(ReadDensity);
                    error.clear(); return true;
                }
                catch (const std::exception& exception)
                {
                    if (Queue.Handle) cudaStreamSynchronize(Queue.Handle);
                    error = exception.what(); return false;
                }
            }
        };
    }
    std::unique_ptr<CudaAirSolver> CreateCudaAirSolver() { return std::make_unique<AirSolver>(); }
}
