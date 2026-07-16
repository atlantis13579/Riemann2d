#pragma once

#include "RHI/RHI.h"

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace Riemann2D::Render
{
    class RenderThread
    {
    public:
        using Factory = std::function<std::unique_ptr<IRHI>()>;

        RenderThread() = default;
        ~RenderThread();

        RenderThread(const RenderThread&) = delete;
        RenderThread& operator=(const RenderThread&) = delete;

        bool Start(Factory factory, const NativeWindow& window);
        bool Submit(SharedFrame frame);
        void Resize(int width, int height);
        bool WaitUntilPresented(std::uint64_t frameId, std::uint32_t timeoutMilliseconds);
        void Stop();

        std::uint64_t GetPresentedFrame() const;
        std::uint64_t GetDroppedFrameCount() const;

    private:
        void ThreadMain(Factory factory, NativeWindow window);

        mutable std::mutex Mutex;
        std::condition_variable Condition;
        std::condition_variable StartedCondition;
        std::condition_variable PresentedCondition;
        std::thread Worker;
        SharedFrame PendingFrame;
        int PendingWidth = 0;
        int PendingHeight = 0;
        std::uint64_t PresentedFrame = 0;
        std::uint64_t DroppedFrames = 0;
        bool Running = false;
        bool Started = false;
        bool InitializationSucceeded = false;
        bool ResizePending = false;
    };
}
