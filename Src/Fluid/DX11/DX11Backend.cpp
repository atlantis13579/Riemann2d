#if defined(_WIN32) && defined(RIEMANN2D_HAS_DX11)
#include "Fluid/FluidComputeBackend.h"
#include "Compute/DX11Compute.h"
namespace Riemann2D
{
    const ComputeShaderBackendInfo &GetComputeShaderBackendInfo()
    {
        static const ComputeShaderBackendInfo Info = []
        {
            ComputeShaderBackendInfo Result;
            try
            {
                const auto& Native = DX11Compute::QueryBackendInfo();
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
