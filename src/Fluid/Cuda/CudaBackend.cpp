#include "CudaSolvers.h"

#if defined(RIEMANN2D_HAS_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace Riemann2D
{
    const CudaBackendInfo& GetCudaBackendInfo()
    {
        static const CudaBackendInfo info = [] {
            CudaBackendInfo result;
#if defined(RIEMANN2D_HAS_CUDA)
            int count = 0;
            const cudaError_t status = cudaGetDeviceCount(&count);
            if (status != cudaSuccess)
            {
                result.Message = cudaGetErrorString(status);
                return result;
            }
            for (int device = 0; device < count; ++device)
            {
                cudaDeviceProp properties{};
                if (cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
                    properties.major * 10 + properties.minor >= 75)
                {
                    result.Available = true;
                    result.DeviceIndex = device;
                    result.DeviceName = properties.name;
                    result.Message = "CUDA ready";
                    return result;
                }
            }
            result.Message = "CUDA requires an NVIDIA GPU with compute capability 7.5 or newer";
#else
            result.Message = "CUDA support was not built";
#endif
            return result;
        }();
        return info;
    }

#if !defined(RIEMANN2D_HAS_CUDA)
    std::unique_ptr<CudaFluidSolver> CreateCudaFluidSolver() { return {}; }
    std::unique_ptr<CudaAirSolver> CreateCudaAirSolver() { return {}; }
#endif
}
