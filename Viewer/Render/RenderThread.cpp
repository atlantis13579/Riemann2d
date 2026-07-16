#include "RenderThread.h"

#include <chrono>
#include <utility>

namespace Riemann2D::Render
{
    RenderThread::~RenderThread()
    {
        Stop();
    }

    bool RenderThread::Start(Factory factory, const NativeWindow& window)
    {
        std::unique_lock<std::mutex> lock(Mutex);
        if (Running || !factory)
        {
            return false;
        }

        Running = true;
        Started = false;
        InitializationSucceeded = false;
        Worker = std::thread(&RenderThread::ThreadMain, this, std::move(factory), window);
        StartedCondition.wait(lock, [this] { return Started; });
        const bool succeeded = InitializationSucceeded;
        lock.unlock();

        if (!succeeded)
        {
            Stop();
        }
        return succeeded;
    }

    bool RenderThread::Submit(SharedFrame frame)
    {
        if (!frame)
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(Mutex);
        if (!Running || !InitializationSucceeded || frame->FrameId <= PresentedFrame)
        {
            return false;
        }
        if (PendingFrame)
        {
            if (frame->FrameId <= PendingFrame->FrameId)
            {
                return false;
            }
            ++DroppedFrames;
        }
        PendingFrame = std::move(frame);
        Condition.notify_one();
        return true;
    }

    void RenderThread::Resize(int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            return;
        }
        std::lock_guard<std::mutex> lock(Mutex);
        PendingWidth = width;
        PendingHeight = height;
        ResizePending = true;
        Condition.notify_one();
    }

    bool RenderThread::WaitUntilPresented(std::uint64_t frameId, std::uint32_t timeoutMilliseconds)
    {
        std::unique_lock<std::mutex> lock(Mutex);
        return PresentedCondition.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
            [this, frameId] { return PresentedFrame >= frameId || !Running; }) && PresentedFrame >= frameId;
    }

    void RenderThread::Stop()
    {
        {
            std::lock_guard<std::mutex> lock(Mutex);
            if (!Worker.joinable())
            {
                Running = false;
                return;
            }
            Running = false;
            PendingFrame.reset();
            Condition.notify_one();
            PresentedCondition.notify_all();
        }

        Worker.join();
    }

    std::uint64_t RenderThread::GetPresentedFrame() const
    {
        std::lock_guard<std::mutex> lock(Mutex);
        return PresentedFrame;
    }

    std::uint64_t RenderThread::GetDroppedFrameCount() const
    {
        std::lock_guard<std::mutex> lock(Mutex);
        return DroppedFrames;
    }

    void RenderThread::ThreadMain(Factory factory, NativeWindow window)
    {
        std::unique_ptr<IRHI> rhi = factory();
        const bool initialized = rhi && rhi->Initialize(window);
        {
            std::lock_guard<std::mutex> lock(Mutex);
            InitializationSucceeded = initialized;
            Started = true;
            if (!initialized)
            {
                Running = false;
            }
            StartedCondition.notify_one();
        }

        if (!initialized)
        {
            return;
        }

        for (;;)
        {
            SharedFrame frame;
            int width = 0;
            int height = 0;
            bool resize = false;
            {
                std::unique_lock<std::mutex> lock(Mutex);
                Condition.wait(lock, [this] { return !Running || PendingFrame || ResizePending; });
                if (!Running)
                {
                    break;
                }
                frame = std::move(PendingFrame);
                if (ResizePending)
                {
                    width = PendingWidth;
                    height = PendingHeight;
                    resize = true;
                    ResizePending = false;
                }
            }

            if (resize)
            {
                rhi->Resize(width, height);
            }
            if (frame)
            {
                rhi->Draw(*frame);
                std::lock_guard<std::mutex> lock(Mutex);
                PresentedFrame = frame->FrameId;
                PresentedCondition.notify_all();
            }
        }

        rhi->Shutdown();
    }
}
