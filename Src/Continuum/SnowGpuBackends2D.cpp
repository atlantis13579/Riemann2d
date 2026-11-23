#include "SnowGpuSolver2D.h"

namespace Riemann2D
{
#if !defined(RIEMANN2D_HAS_CUDA)
    std::unique_ptr<SnowGpuSolver2D> CreateCudaSnowSolver2D() { return {}; }
#endif
}
