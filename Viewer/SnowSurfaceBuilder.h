#pragma once

#include "Continuum/SnowSimulation2D.h"
#include "Render/RenderTypes.h"

namespace Riemann2D::Viewer
{
    Render::DrawPacket BuildSnowSurfaceDraw(const SnowSimulation2D& simulation, ContinuumQuality quality);
}
