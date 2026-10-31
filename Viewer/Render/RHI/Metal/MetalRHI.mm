#include "RHI/Metal/MetalRHI.h"

#include "Projection.h"
#include "FieldShading.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <vector>
#include <string>

namespace Riemann2D::Render
{
    namespace
    {
        constexpr const char* ShaderSource = R"(
            #include <metal_stdlib>
            using namespace metal;
            struct VertexIn { packed_float2 position; packed_float4 color; packed_float2 uv; };
            struct VertexOut { float4 position [[position]]; float4 color; float2 uv; };
            vertex VertexOut vertexMain(const device VertexIn* vertices [[buffer(0)]],
                                        uint vertexId [[vertex_id]]) {
                VertexOut output;
                output.position = float4(vertices[vertexId].position, 0.0, 1.0);
                output.color = vertices[vertexId].color;
                output.uv = vertices[vertexId].uv;
                return output;
            }
            fragment float4 fragmentMain(VertexOut input [[stage_in]]) { return input.color; }
            #define lerp mix
            #define SAMPLE_FIELD(coordinate) field.sample(fieldSampler, coordinate).r
            fragment float4 fragmentField(VertexOut input [[stage_in]],
                texture2d<float> field [[texture(0)]], sampler fieldSampler [[sampler(0)]],
                constant float4& parameters [[buffer(0)]]) {
                float2 uv = input.uv;
        )";

        struct GPUVertex
        {
            float X;
            float Y;
            float R;
            float G;
            float B;
            float A;
            float U;
            float V;
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
            id<MTLRenderPipelineState> FieldPipeline = nil;
            id<MTLSamplerState> FieldSampler = nil;
            NSMutableArray<id<MTLTexture>>* FieldTextures[3] = {nil, nil, nil};
            dispatch_semaphore_t Inflight = nullptr;
            int FrameSlot = 0;
            NSUInteger FieldIndex = 0;
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
            const std::string shader = std::string(ShaderSource) + FieldShading + "}";
            NSString* source = [NSString stringWithUTF8String:shader.c_str()];
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
            descriptor.fragmentFunction = [library newFunctionWithName:@"fragmentField"];
            FieldPipeline = [Device newRenderPipelineStateWithDescriptor:descriptor error:&error];
            MTLSamplerDescriptor* sampler = [[MTLSamplerDescriptor alloc] init];
            sampler.minFilter = sampler.magFilter = MTLSamplerMinMagFilterLinear;
            sampler.sAddressMode = sampler.tAddressMode = MTLSamplerAddressModeClampToEdge;
            FieldSampler = [Device newSamplerStateWithDescriptor:sampler];
            Inflight = dispatch_semaphore_create(3);
            for (auto& textures : FieldTextures) textures = [[NSMutableArray alloc] init];
            return CommandQueue && Pipeline && FieldPipeline && FieldSampler;
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
                                        source.Tint.B, source.Tint.A, source.TexCoord.X, source.TexCoord.Y});
                }
                id<MTLBuffer> vertexBuffer = [Device newBufferWithBytes:vertices.data()
                    length:vertices.size() * sizeof(GPUVertex) options:MTLResourceStorageModeShared];
                id<MTLBuffer> indexBuffer = [Device newBufferWithBytes:packet.Indices.data()
                    length:packet.Indices.size() * sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
                [encoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
                const MTLPrimitiveType topology = packet.Topology == PrimitiveTopology::Lines
                    ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;
                if (packet.Field)
                {
                    const ScalarField& field = *packet.Field;
                    if (!field.IsValid()) continue;
                    auto* textures = FieldTextures[FrameSlot];
                    id<MTLTexture> texture = FieldIndex < textures.count ? textures[FieldIndex] : nil;
                    if (!texture || texture.width != static_cast<NSUInteger>(field.Width) ||
                        texture.height != static_cast<NSUInteger>(field.Height))
                    {
                        MTLTextureDescriptor* description = [MTLTextureDescriptor
                            texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                            width:field.Width height:field.Height mipmapped:NO];
                        description.usage = MTLTextureUsageShaderRead;
                        description.storageMode = MTLStorageModeShared;
                        texture = [Device newTextureWithDescriptor:description];
                        if (!texture) continue;
                        if (FieldIndex < textures.count) textures[FieldIndex] = texture;
                        else [textures addObject:texture];
                    }
                    ++FieldIndex;
                    [texture replaceRegion:MTLRegionMake2D(0, 0, field.Width, field.Height) mipmapLevel:0
                        withBytes:field.Values.data() bytesPerRow:field.Width * sizeof(float)];
                    const float values[4] = {1.0f / field.Width, 1.0f / field.Height,
                        static_cast<float>(field.Material), field.Rotation};
                    [encoder setRenderPipelineState:FieldPipeline];
                    [encoder setFragmentTexture:texture atIndex:0];
                    [encoder setFragmentSamplerState:FieldSampler atIndex:0];
                    [encoder setFragmentBytes:values length:sizeof(values) atIndex:0];
                }
                else [encoder setRenderPipelineState:Pipeline];
                [encoder drawIndexedPrimitives:topology indexCount:packet.Indices.size()
                    indexType:MTLIndexTypeUInt32 indexBuffer:indexBuffer indexBufferOffset:0];
            }
        }

        void MetalRHI::Draw(const FramePacket& frame)
        {
            @autoreleasepool
            {
                dispatch_semaphore_wait(Inflight, DISPATCH_TIME_FOREVER);
                id<CAMetalDrawable> drawable = [Layer nextDrawable];
                if (!drawable)
                {
                    dispatch_semaphore_signal(Inflight);
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
                FieldIndex = 0;
                DrawPackets(encoder, frame.WorldDraws, frame, false);
                DrawPackets(encoder, frame.UserInterfaceDraws, frame, true);
                [encoder endEncoding];
                [commandBuffer presentDrawable:drawable];
                dispatch_semaphore_t semaphore = Inflight;
                [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) { dispatch_semaphore_signal(semaphore); }];
                [commandBuffer commit];
                FrameSlot = (FrameSlot + 1) % 3;
            }
        }

        void MetalRHI::Shutdown()
        {
            if (Inflight)
            {
                for (int index = 0; index < 3; ++index) dispatch_semaphore_wait(Inflight, DISPATCH_TIME_FOREVER);
                Inflight = nullptr;
            }
            for (auto& textures : FieldTextures) textures = nil;
            FieldSampler = nil; FieldPipeline = nil;
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
