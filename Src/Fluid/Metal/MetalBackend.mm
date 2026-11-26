#if defined(RIEMANN2D_HAS_METAL)
#include "Fluid/FluidComputeBackend.h"
#include "Compute/MetalCompute.h"
namespace Riemann2D
{
    const ComputeShaderBackendInfo &GetMetalBackendInfo()
    {
        static const ComputeShaderBackendInfo Info = []
        {
            ComputeShaderBackendInfo Result;
            try
            {
                const auto& Native = MetalCompute::QueryBackendInfo();
                Result.Available = Native.Available;
                Result.API = Native.API;
                Result.DeviceName = Native.DeviceName;
                Result.Message = Native.Message;
            }
            catch (const std::exception &Error)
            {
                Result.Message = Error.what();
            }
            return Result;
        }();
        return Info;
    }
} // namespace Riemann2D
#endif
