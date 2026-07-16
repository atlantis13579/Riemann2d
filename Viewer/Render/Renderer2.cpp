#include "Renderer2.h"

namespace Riemann2D::Render
{
    bool Renderer2::Initialize(RenderThread::Factory factory, const NativeWindow& window)
    {
        return Thread.Start(std::move(factory), window);
    }

    bool Renderer2::Submit(SharedFrame frame)
    {
        return Thread.Submit(std::move(frame));
    }

    void Renderer2::Resize(int width, int height)
    {
        Thread.Resize(width, height);
    }

    void Renderer2::WaitIdle(std::uint64_t lastFrameId)
    {
        Thread.WaitUntilPresented(lastFrameId, 5000);
    }

    void Renderer2::Shutdown()
    {
        Thread.Stop();
    }

    std::uint64_t Renderer2::GetPresentedFrame() const
    {
        return Thread.GetPresentedFrame();
    }
}
