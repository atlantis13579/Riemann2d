#pragma once

#include "RHI/RHI.h"

#include <atomic>
#include <memory>

namespace Riemann2D::Render
{
    struct NullRHIStats
    {
        std::atomic<std::uint64_t> PresentedFrame{0};
        std::atomic<std::uint64_t> DrawPacketCount{0};
        std::atomic<int> Width{0};
        std::atomic<int> Height{0};
        std::atomic<bool> Initialized{false};
        std::atomic<bool> ShutdownCalled{false};
    };

    std::unique_ptr<IRHI> CreateNullRHI();
    std::unique_ptr<IRHI> CreateNullRHI(std::shared_ptr<NullRHIStats> stats);
}
