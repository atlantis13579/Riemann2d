#pragma once

#include <memory>
#include <string>

namespace Riemann2D
{
    enum class FluidComputeBackend { CPU, CUDA, ComputeShader, Metal };

    struct CudaBackendInfo
    {
        bool Available = false;
        int DeviceIndex = -1;
        std::string DeviceName;
        std::string Message;
    };

    const CudaBackendInfo& GetCudaBackendInfo();

    struct ComputeShaderBackendInfo
    {
        bool Available = false;
        std::string DeviceName;
        std::string API;
        std::string Message;
    };

    const ComputeShaderBackendInfo& GetComputeShaderBackendInfo();
    const ComputeShaderBackendInfo& GetMetalBackendInfo();

    enum class GpuOperator { Liquid, Air, Snow };
    struct OperatorAvailability { bool Available = false; std::string Message; };
    // Availability describes build/device support; compilation and execution may still fail at use.
    const OperatorAvailability& GetOperatorAvailability(GpuOperator operation, FluidComputeBackend backend);
    void ProbeGpuOperators();

    // Simulation copies retain host state but never share mutable device resources.
    template<class Backend>
    struct ComputeResource
    {
        std::unique_ptr<Backend> Instance;
        ComputeResource() = default;
        ComputeResource(const ComputeResource&) {}
        ComputeResource& operator=(const ComputeResource&) { Instance.reset(); return *this; }
        ComputeResource(ComputeResource&&) noexcept = default;
        ComputeResource& operator=(ComputeResource&&) noexcept = default;
    };
}
