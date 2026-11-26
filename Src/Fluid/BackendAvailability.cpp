#include "FluidComputeBackend.h"
#include "Cuda/CudaSolvers.h"
#include "Continuum/SnowGpuSolver2D.h"
#include <array>
#include <exception>

namespace Riemann2D
{
#if !defined(_WIN32) || !defined(RIEMANN2D_HAS_DX11)
    std::unique_ptr<FluidGpuSolver> CreateDX11FluidSolver()
    {
        return {};
    }
    std::unique_ptr<AirGpuSolver> CreateDX11AirSolver()
    {
        return {};
    }
    std::unique_ptr<SnowGpuSolver2D> CreateDX11SnowSolver2D()
    {
        return {};
    }
    const ComputeShaderBackendInfo &GetComputeShaderBackendInfo()
    {
        static const ComputeShaderBackendInfo Info{false, "", "DX11", "DirectCompute support was not built"};
        return Info;
    }
#endif
#if !defined(RIEMANN2D_HAS_METAL)
    std::unique_ptr<FluidGpuSolver> CreateMetalFluidSolver()
    {
        return {};
    }
    std::unique_ptr<AirGpuSolver> CreateMetalAirSolver()
    {
        return {};
    }
    std::unique_ptr<SnowGpuSolver2D> CreateMetalSnowSolver2D()
    {
        return {};
    }
    const ComputeShaderBackendInfo &GetMetalBackendInfo()
    {
        static const ComputeShaderBackendInfo Info{false, "", "Metal", "Metal support was not built"};
        return Info;
    }
#endif
    namespace
    {
        using Capabilities = std::array<std::array<OperatorAvailability, 4>, 3>;
        Capabilities Probe()
        {
            Capabilities Result;
            // These three operators are compiled together for each native backend.
            for (int Backend = 0; Backend < 4; ++Backend)
            {
                OperatorAvailability Entry;
                try
                {
                    if (Backend == 0) Entry = {true, "CPU available"};
                    else if (Backend == 1)
                    {
                        const auto& Info = GetCudaBackendInfo();
                        Entry = {Info.Available, Info.Message};
                    }
                    else
                    {
                        const auto& Info = Backend == 2 ? GetComputeShaderBackendInfo() : GetMetalBackendInfo();
                        Entry = {Info.Available, Info.Message};
                    }
                }
                catch (const std::exception& Error)
                {
                    Entry = {false, Error.what()};
                }
                for (auto& Operation : Result) Operation[Backend] = Entry;
            }
            return Result;
        }
        const Capabilities &GetCapabilities()
        {
            static const Capabilities Value = Probe();
            return Value;
        }
    } // namespace
    void ProbeGpuOperators()
    {
        (void)GetCapabilities();
    }
    const OperatorAvailability &GetOperatorAvailability(GpuOperator Operation, FluidComputeBackend Backend)
    {
        const int O = static_cast<int>(Operation), B = static_cast<int>(Backend);
        static const OperatorAvailability Invalid{false, "Unknown operator or backend"};
        static const OperatorAvailability CPU{true, "CPU available"};
        if (O < 0 || O >= 3 || B < 0 || B >= 4) return Invalid;
        return B == 0 ? CPU : GetCapabilities()[O][B];
    }

} // namespace Riemann2D
