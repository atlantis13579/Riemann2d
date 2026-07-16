#include "RHI/Null/NullRHI.h"

namespace Riemann2D::Render
{
    namespace
    {
        class NullRHI final : public IRHI
        {
        public:
            explicit NullRHI(std::shared_ptr<NullRHIStats> stats)
                : Stats(std::move(stats))
            {
            }

            bool Initialize(const NativeWindow& window) override
            {
                Stats->Width = window.Width;
                Stats->Height = window.Height;
                Stats->Initialized = true;
                return true;
            }

            void Resize(int width, int height) override
            {
                Stats->Width = width;
                Stats->Height = height;
            }

            void Draw(const FramePacket& frame) override
            {
                Stats->DrawPacketCount = frame.WorldDraws.size() + frame.UserInterfaceDraws.size();
                Stats->PresentedFrame = frame.FrameId;
            }

            void Shutdown() override
            {
                Stats->ShutdownCalled = true;
                Stats->Initialized = false;
            }

        private:
            std::shared_ptr<NullRHIStats> Stats;
        };
    }

    std::unique_ptr<IRHI> CreateNullRHI()
    {
        return CreateNullRHI(std::make_shared<NullRHIStats>());
    }

    std::unique_ptr<IRHI> CreateNullRHI(std::shared_ptr<NullRHIStats> stats)
    {
        return std::make_unique<NullRHI>(std::move(stats));
    }
}
