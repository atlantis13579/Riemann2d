#pragma once

#include "RHI/RHI.h"

namespace Riemann2D::Render
{
    std::unique_ptr<IRHI> CreateMetalRHI();
}
