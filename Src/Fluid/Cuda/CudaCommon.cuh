#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace Riemann2D::Cuda
{
    inline void Check(cudaError_t status)
    {
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }

    template<class T>
    class Buffer
    {
    public:
        T* Data = nullptr;
        std::size_t Capacity = 0;
        Buffer() = default;
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&& other) noexcept { std::swap(Data, other.Data); std::swap(Capacity, other.Capacity); }
        Buffer& operator=(Buffer&& other) noexcept
        {
            std::swap(Data, other.Data); std::swap(Capacity, other.Capacity); return *this;
        }
        ~Buffer() { if (Data) cudaFree(Data); }
        void Reserve(std::size_t count)
        {
            if (count <= Capacity) return;
            T* replacement = nullptr;
            Check(cudaMalloc(reinterpret_cast<void**>(&replacement), count * sizeof(T)));
            if (Data) cudaFree(Data);
            Data = replacement; Capacity = count;
        }
    };

    struct Stream
    {
        cudaStream_t Handle = nullptr;
        ~Stream() { if (Handle) cudaStreamDestroy(Handle); }
        void Initialize()
        {
            if (!Handle) Check(cudaStreamCreateWithFlags(&Handle, cudaStreamNonBlocking));
        }
    };

    struct Float2 { float X, Y; };
    __host__ __device__ inline Float2 operator+(Float2 a, Float2 b) { return {a.X + b.X, a.Y + b.Y}; }
    __host__ __device__ inline Float2 operator-(Float2 a, Float2 b) { return {a.X - b.X, a.Y - b.Y}; }
    __host__ __device__ inline Float2 operator*(float a, Float2 b) { return {a * b.X, a * b.Y}; }
    __host__ __device__ inline float Dot(Float2 a, Float2 b) { return a.X * b.X + a.Y * b.Y; }
    __device__ inline float Clamp(float x, float a, float b) { return fminf(fmaxf(x, a), b); }
    inline int Blocks(int count) { return (count + 255) / 256; }
}
