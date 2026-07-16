#pragma once

#include "RenderThread.h"

namespace Riemann2D::Render
{
    class Renderer2
    {
    public:
        bool Initialize(RenderThread::Factory factory, const NativeWindow& window);
        bool Submit(SharedFrame frame);
        void Resize(int width, int height);
        void WaitIdle(std::uint64_t lastFrameId);
        void Shutdown();

        std::uint64_t GetPresentedFrame() const;

    private:
        RenderThread Thread;
    };
}
