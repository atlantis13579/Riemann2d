#include "RHI/Metal/MetalRHI.h"

#include "Projection.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <vector>

namespace Riemann2D::Render
{
    namespace
    {
        constexpr const char* ShaderSource = R"(
            #include <metal_stdlib>
            using namespace metal;
            struct VertexIn { packed_float2 position; packed_float4 color; };
            struct VertexOut { float4 position [[position]]; float4 color; };
            vertex VertexOut vertexMain(const device VertexIn* vertices [[buffer(0)]],
                                        uint vertexId [[vertex_id]]) {
                VertexOut output;
                output.position = float4(vertices[vertexId].position, 0.0, 1.0);
                output.color = vertices[vertexId].color;
                return output;
            }
            fragment float4 fragmentMain(VertexOut input [[stage_in]]) { return input.color; }
        )";

        struct GPUVertex
        {
            float X;
            float Y;
            float R;
            float G;
            float B;
            float A;
        };

        class MetalRHI final : public IRHI
        {
        public:
            bool Initialize(const NativeWindow& window) override;
            void Resize(int width, int height) override;
            void Draw(const FramePacket& frame) override;
            void Shutdown() override;

        private:
            void DrawPackets(id<MTLRenderCommandEncoder> encoder, const std::vector<DrawPacket>& packets,
                             const FramePacket& frame, bool interfaceSpace);

            CAMetalLayer* Layer = nil;
            id<MTLDevice> Device = nil;
            id<MTLCommandQueue> CommandQueue = nil;
            id<MTLRenderPipelineState> Pipeline = nil;
        };

        bool MetalRHI::Initialize(const NativeWindow& window)
        {
            if (window.Platform != WindowPlatform::Cocoa || !window.Handle)
            {
                return false;
            }
            Layer = (__bridge CAMetalLayer*)window.Handle;
            Device = MTLCreateSystemDefaultDevice();
            if (!Device)
            {
                return false;
            }
            Layer.device = Device;
            Layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
            Layer.framebufferOnly = YES;
            Layer.drawableSize = CGSizeMake(window.Width, window.Height);
            CommandQueue = [Device newCommandQueue];

            NSError* error = nil;
            NSString* source = [NSString stringWithUTF8String:ShaderSource];
            id<MTLLibrary> library = [Device newLibraryWithSource:source options:nil error:&error];
            if (!library)
            {
                return false;
            }
            MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
            descriptor.vertexFunction = [library newFunctionWithName:@"vertexMain"];
            descriptor.fragmentFunction = [library newFunctionWithName:@"fragmentMain"];
            descriptor.colorAttachments[0].pixelFormat = Layer.pixelFormat;
            descriptor.colorAttachments[0].blendingEnabled = YES;
            descriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            descriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            Pipeline = [Device newRenderPipelineStateWithDescriptor:descriptor error:&error];
            return CommandQueue && Pipeline;
        }

        void MetalRHI::Resize(int width, int height)
        {
            if (width > 0 && height > 0)
            {
                Layer.drawableSize = CGSizeMake(width, height);
            }
        }

        void MetalRHI::DrawPackets(id<MTLRenderCommandEncoder> encoder,
                                   const std::vector<DrawPacket>& packets,
                                   const FramePacket& frame, bool interfaceSpace)
        {
            for (const DrawPacket& packet : packets)
            {
                if (packet.Vertices.empty() || packet.Indices.empty())
                {
                    continue;
                }
                std::vector<GPUVertex> vertices;
                vertices.reserve(packet.Vertices.size());
                for (const Vertex& source : packet.Vertices)
                {
                    const Vector2 clip = interfaceSpace ? ProjectInterfaceToClip(source.Position, frame)
                                                        : ProjectWorldToClip(source.Position, frame);
                    vertices.push_back({clip.X, clip.Y, source.Tint.R, source.Tint.G,
                                        source.Tint.B, source.Tint.A});
                }
                id<MTLBuffer> vertexBuffer = [Device newBufferWithBytes:vertices.data()
                    length:vertices.size() * sizeof(GPUVertex) options:MTLResourceStorageModeShared];
                id<MTLBuffer> indexBuffer = [Device newBufferWithBytes:packet.Indices.data()
                    length:packet.Indices.size() * sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
                [encoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
                const MTLPrimitiveType topology = packet.Topology == PrimitiveTopology::Lines
                    ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;
                [encoder drawIndexedPrimitives:topology indexCount:packet.Indices.size()
                    indexType:MTLIndexTypeUInt32 indexBuffer:indexBuffer indexBufferOffset:0];
            }
        }

        void MetalRHI::Draw(const FramePacket& frame)
        {
            @autoreleasepool
            {
                id<CAMetalDrawable> drawable = [Layer nextDrawable];
                if (!drawable)
                {
                    return;
                }
                MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                pass.colorAttachments[0].texture = drawable.texture;
                pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                pass.colorAttachments[0].clearColor = MTLClearColorMake(
                    frame.ClearColor.R, frame.ClearColor.G, frame.ClearColor.B, frame.ClearColor.A);
                id<MTLCommandBuffer> commandBuffer = [CommandQueue commandBuffer];
                id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
                [encoder setRenderPipelineState:Pipeline];
                [encoder setCullMode:MTLCullModeNone];
                DrawPackets(encoder, frame.WorldDraws, frame, false);
                DrawPackets(encoder, frame.UserInterfaceDraws, frame, true);
                [encoder endEncoding];
                [commandBuffer presentDrawable:drawable];
                [commandBuffer commit];
            }
        }

        void MetalRHI::Shutdown()
        {
            Pipeline = nil;
            CommandQueue = nil;
            Device = nil;
            Layer = nil;
        }
    }

    std::unique_ptr<IRHI> CreateMetalRHI()
    {
        return std::make_unique<MetalRHI>();
    }
}
