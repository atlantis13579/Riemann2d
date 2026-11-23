#include "ComputeDevice.h"

#include <exception>

namespace Riemann2D
{
    const ComputeShaderBackendInfo& GetComputeShaderBackendInfo()
    {
        static const ComputeShaderBackendInfo info = [] {
            ComputeShaderBackendInfo result;
            try
            {
                auto device = Compute::CreateDevice();
                if (device) return device->GetInfo();
                result.Message = "Compute Shader support was not built";
            }
            catch (const std::exception& error) { result.Message = error.what(); }
            return result;
        }();
        return info;
    }

    namespace Compute
    {
        void Reserve(Device& device, SharedBuffer& buffer, std::size_t bytes)
        {
            if (!buffer || buffer->Size < bytes) buffer = device.CreateBuffer(bytes);
        }

#if !defined(RIEMANN2D_HAS_COMPUTE_SHADER)
        std::unique_ptr<Device> CreateDevice() { return {}; }
#endif
    }
}
