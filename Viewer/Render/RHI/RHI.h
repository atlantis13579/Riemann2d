#pragma once

#include "RenderTypes.h"

#include <memory>

namespace Riemann2D::Render
{
    enum class WindowPlatform
    {
        None,
        Win32,
        Cocoa
    };

    struct NativeWindow
    {
        WindowPlatform Platform = WindowPlatform::None;
        void* Handle = nullptr;
        int Width = 1280;
        int Height = 720;
    };

    class IRHI
    {
    public:
        virtual ~IRHI() = default;

        // Every method is called exclusively by the render thread.
        virtual bool Initialize(const NativeWindow& window) = 0;
        virtual void Resize(int width, int height) = 0;
        virtual void Draw(const FramePacket& frame) = 0;
        virtual void Shutdown() = 0;
    };

    using RHIFactory = std::unique_ptr<IRHI> (*)();
}
