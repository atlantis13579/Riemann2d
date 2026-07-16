#include "RHI/PlatformRHI.h"

#if defined(_WIN32)
#include "RHI/DX11/DX11RHI.h"
#elif defined(__APPLE__)
#include "RHI/Metal/MetalRHI.h"
#endif

namespace Riemann2D::Render
{
    std::unique_ptr<IRHI> CreatePlatformRHI()
    {
#if defined(_WIN32)
        return CreateDX11RHI();
#elif defined(__APPLE__)
        return CreateMetalRHI();
#else
        return nullptr;
#endif
    }
}
