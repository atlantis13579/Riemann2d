#pragma once

#include "Fluid/FluidSimulation2D.h"
#include "Render/RenderTypes.h"

namespace Riemann2D::Viewer
{
    struct FluidSurfaceDraw
    {
        Render::DrawPacket Fill;
        Render::DrawPacket Outline;
    };

    FluidSurfaceDraw BuildFluidSurfaceDraw(const FluidSimulation2D& fluid);
}
