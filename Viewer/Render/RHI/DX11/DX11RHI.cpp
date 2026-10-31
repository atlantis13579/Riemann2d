#include "RHI/DX11/DX11RHI.h"

#include "Projection.h"
#include "FieldShading.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <algorithm>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <cstdio>

namespace Riemann2D::Render
{
    using Microsoft::WRL::ComPtr;

    namespace
    {
        constexpr char ShaderSource[] = R"(
            struct VSInput { float2 Position : POSITION; float4 Color : COLOR; float2 UV : TEXCOORD; };
            struct PSInput { float4 Position : SV_POSITION; float4 Color : COLOR; float2 UV : TEXCOORD; };
            PSInput VSMain(VSInput input) {
                PSInput output;
                output.Position = float4(input.Position, 0.0, 1.0);
                output.Color = input.Color;
                output.UV = input.UV;
                return output;
            }
            float4 PSMain(PSInput input) : SV_TARGET { return input.Color; }
            Texture2D<float> ScalarTexture : register(t0);
            SamplerState LinearSampler : register(s0);
            cbuffer FieldConstants : register(b0) { float4 parameters; };
            #define SAMPLE_FIELD(coordinate) ScalarTexture.Sample(LinearSampler, coordinate)
            float4 PSField(PSInput input) : SV_TARGET {
                float2 uv = input.UV;
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

        class DX11RHI final : public IRHI
        {
        public:
            bool Initialize(const NativeWindow& window) override;
            void Resize(int width, int height) override;
            void Draw(const FramePacket& frame) override;
            void Shutdown() override;

        private:
            bool CreateRenderTarget();
            bool CreatePipeline();
            bool BindField(const ScalarField& field);
            void Capture(const std::string& path);
            bool EnsureBuffers(std::size_t vertexCount, std::size_t indexCount);
            void DrawPackets(const std::vector<DrawPacket>& packets, const FramePacket& frame, bool interfaceSpace);

            ComPtr<ID3D11Device> Device;
            ComPtr<ID3D11DeviceContext> Context;
            ComPtr<IDXGISwapChain> SwapChain;
            ComPtr<ID3D11RenderTargetView> RenderTarget;
            ComPtr<ID3D11VertexShader> VertexShader;
            ComPtr<ID3D11PixelShader> PixelShader;
            ComPtr<ID3D11PixelShader> FieldPixelShader;
            ComPtr<ID3D11Texture2D> FieldTexture;
            ComPtr<ID3D11ShaderResourceView> FieldView;
            ComPtr<ID3D11SamplerState> FieldSampler;
            ComPtr<ID3D11Buffer> FieldConstants;
            int FieldWidth = 0;
            int FieldHeight = 0;
            ComPtr<ID3D11InputLayout> InputLayout;
            ComPtr<ID3D11BlendState> BlendState;
            ComPtr<ID3D11RasterizerState> RasterizerState;
            ComPtr<ID3D11Buffer> VertexBuffer;
            ComPtr<ID3D11Buffer> IndexBuffer;
            std::size_t VertexCapacity = 0;
            std::size_t IndexCapacity = 0;
            int Width = 0;
            int Height = 0;
        };

        bool DX11RHI::Initialize(const NativeWindow& window)
        {
            if (window.Platform != WindowPlatform::Win32 || !window.Handle)
            {
                return false;
            }
            Width = window.Width;
            Height = window.Height;
            DXGI_SWAP_CHAIN_DESC description{};
            description.BufferDesc.Width = static_cast<UINT>(Width);
            description.BufferDesc.Height = static_cast<UINT>(Height);
            description.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            description.SampleDesc.Count = 1;
            description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            description.BufferCount = 2;
            description.OutputWindow = static_cast<HWND>(window.Handle);
            description.Windowed = TRUE;
            description.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

            const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
            const D3D_FEATURE_LEVEL fallbackLevels[] = {D3D_FEATURE_LEVEL_11_0};
            D3D_FEATURE_LEVEL selected{};
            auto createDevice = [&](D3D_DRIVER_TYPE driverType, const D3D_FEATURE_LEVEL* featureLevels,
                                    UINT featureLevelCount)
            {
                return D3D11CreateDeviceAndSwapChain(
                    nullptr, driverType, nullptr, 0, featureLevels, featureLevelCount, D3D11_SDK_VERSION,
                    &description, &SwapChain, &Device, &selected, &Context);
            };

            HRESULT result = createDevice(D3D_DRIVER_TYPE_HARDWARE, levels, 2);
            if (result == E_INVALIDARG)
            {
                result = createDevice(D3D_DRIVER_TYPE_HARDWARE, fallbackLevels, 1);
            }
            if (FAILED(result))
            {
                result = createDevice(D3D_DRIVER_TYPE_WARP, fallbackLevels, 1);
            }
            (void)selected;
            return SUCCEEDED(result) && CreateRenderTarget() && CreatePipeline();
        }

        bool DX11RHI::CreateRenderTarget()
        {
            ComPtr<ID3D11Texture2D> backBuffer;
            if (FAILED(SwapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer))))
            {
                return false;
            }
            return SUCCEEDED(Device->CreateRenderTargetView(backBuffer.Get(), nullptr, &RenderTarget));
        }

        bool DX11RHI::CreatePipeline()
        {
            const std::string shader = std::string(ShaderSource) + FieldShading + "}";
            ComPtr<ID3DBlob> fieldCode;
            ComPtr<ID3DBlob> vertexCode;
            ComPtr<ID3DBlob> pixelCode;
            ComPtr<ID3DBlob> errors;
            if (FAILED(D3DCompile(shader.data(), shader.size(), nullptr, nullptr, nullptr, "VSMain", "vs_5_0",
                                  0, 0, &vertexCode, &errors)) ||
                FAILED(D3DCompile(shader.data(), shader.size(), nullptr, nullptr, nullptr, "PSMain", "ps_5_0",
                                  0, 0, &pixelCode, &errors)) ||
                FAILED(D3DCompile(shader.data(), shader.size(), nullptr, nullptr, nullptr, "PSField", "ps_5_0",
                                  D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &fieldCode, &errors)))
            {
                if (errors) std::fputs(static_cast<const char*>(errors->GetBufferPointer()), stderr);
                return false;
            }
            if (FAILED(Device->CreateVertexShader(vertexCode->GetBufferPointer(), vertexCode->GetBufferSize(),
                                                  nullptr, &VertexShader)) ||
                FAILED(Device->CreatePixelShader(pixelCode->GetBufferPointer(), pixelCode->GetBufferSize(),
                                                 nullptr, &PixelShader)) ||
                FAILED(Device->CreatePixelShader(fieldCode->GetBufferPointer(), fieldCode->GetBufferSize(),
                                                 nullptr, &FieldPixelShader)))
            {
                return false;
            }
            const D3D11_INPUT_ELEMENT_DESC elements[] = {
                {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
                {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0},
                {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0}
            };
            if (FAILED(Device->CreateInputLayout(elements, 3, vertexCode->GetBufferPointer(),
                                                 vertexCode->GetBufferSize(), &InputLayout)))
            {
                return false;
            }
            D3D11_SAMPLER_DESC sampler{};
            sampler.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
            sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            sampler.MaxLOD = D3D11_FLOAT32_MAX;
            if (FAILED(Device->CreateSamplerState(&sampler, &FieldSampler))) return false;
            D3D11_BUFFER_DESC constants{};
            constants.ByteWidth = 16; constants.Usage = D3D11_USAGE_DYNAMIC;
            constants.BindFlags = D3D11_BIND_CONSTANT_BUFFER; constants.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (FAILED(Device->CreateBuffer(&constants, nullptr, &FieldConstants))) return false;
            D3D11_BLEND_DESC blend{};
            blend.RenderTarget[0].BlendEnable = TRUE;
            blend.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
            blend.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
            blend.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
            blend.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
            blend.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
            blend.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
            blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
            if (FAILED(Device->CreateBlendState(&blend, &BlendState)))
            {
                return false;
            }

            D3D11_RASTERIZER_DESC rasterizer{};
            rasterizer.FillMode = D3D11_FILL_SOLID;
            rasterizer.CullMode = D3D11_CULL_NONE;
            rasterizer.DepthClipEnable = TRUE;
            return SUCCEEDED(Device->CreateRasterizerState(&rasterizer, &RasterizerState));
        }

        void DX11RHI::Resize(int width, int height)
        {
            if (width <= 0 || height <= 0 || !SwapChain)
            {
                return;
            }
            Context->OMSetRenderTargets(0, nullptr, nullptr);
            RenderTarget.Reset();
            Width = width;
            Height = height;
            if (SUCCEEDED(SwapChain->ResizeBuffers(0, static_cast<UINT>(width), static_cast<UINT>(height),
                                                  DXGI_FORMAT_UNKNOWN, 0)))
            {
                CreateRenderTarget();
            }
        }

        bool DX11RHI::BindField(const ScalarField& field)
        {
            if (!field.IsValid()) return false;
            ID3D11ShaderResourceView* empty = nullptr;
            Context->PSSetShaderResources(0, 1, &empty);
            if (!FieldTexture || FieldWidth != field.Width || FieldHeight != field.Height)
            {
                FieldView.Reset(); FieldTexture.Reset();
                D3D11_TEXTURE2D_DESC description{};
                description.Width = static_cast<UINT>(field.Width); description.Height = static_cast<UINT>(field.Height);
                description.MipLevels = 1; description.ArraySize = 1; description.SampleDesc.Count = 1;
                description.Format = DXGI_FORMAT_R32_FLOAT; description.Usage = D3D11_USAGE_DYNAMIC;
                description.BindFlags = D3D11_BIND_SHADER_RESOURCE; description.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                if (FAILED(Device->CreateTexture2D(&description, nullptr, &FieldTexture)) ||
                    FAILED(Device->CreateShaderResourceView(FieldTexture.Get(), nullptr, &FieldView))) return false;
                FieldWidth = field.Width; FieldHeight = field.Height;
            }
            D3D11_MAPPED_SUBRESOURCE mapping{};
            if (FAILED(Context->Map(FieldTexture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapping))) return false;
            for (int y = 0; y < field.Height; ++y)
                std::memcpy(static_cast<char*>(mapping.pData) + y * mapping.RowPitch,
                            field.Values.data() + y * field.Width, field.Width * sizeof(float));
            Context->Unmap(FieldTexture.Get(), 0);
            if (FAILED(Context->Map(FieldConstants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapping))) return false;
            const float values[4] = {1.0f / field.Width, 1.0f / field.Height,
                                    static_cast<float>(field.Material), field.Rotation};
            std::memcpy(mapping.pData, values, sizeof(values));
            Context->Unmap(FieldConstants.Get(), 0);
            Context->PSSetShader(FieldPixelShader.Get(), nullptr, 0);
            Context->PSSetShaderResources(0, 1, FieldView.GetAddressOf());
            Context->PSSetSamplers(0, 1, FieldSampler.GetAddressOf());
            Context->PSSetConstantBuffers(0, 1, FieldConstants.GetAddressOf());
            return true;
        }

        bool DX11RHI::EnsureBuffers(std::size_t vertexCount, std::size_t indexCount)
        {
            if (vertexCount > VertexCapacity)
            {
                VertexCapacity = std::max(vertexCount, VertexCapacity * 2 + 256);
                D3D11_BUFFER_DESC description{};
                description.ByteWidth = static_cast<UINT>(VertexCapacity * sizeof(GPUVertex));
                description.Usage = D3D11_USAGE_DYNAMIC;
                description.BindFlags = D3D11_BIND_VERTEX_BUFFER;
                description.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                VertexBuffer.Reset();
                if (FAILED(Device->CreateBuffer(&description, nullptr, &VertexBuffer))) return false;
            }
            if (indexCount > IndexCapacity)
            {
                IndexCapacity = std::max(indexCount, IndexCapacity * 2 + 512);
                D3D11_BUFFER_DESC description{};
                description.ByteWidth = static_cast<UINT>(IndexCapacity * sizeof(std::uint32_t));
                description.Usage = D3D11_USAGE_DYNAMIC;
                description.BindFlags = D3D11_BIND_INDEX_BUFFER;
                description.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                IndexBuffer.Reset();
                if (FAILED(Device->CreateBuffer(&description, nullptr, &IndexBuffer))) return false;
            }
            return true;
        }

        void DX11RHI::DrawPackets(const std::vector<DrawPacket>& packets, const FramePacket& frame, bool interfaceSpace)
        {
            for (const DrawPacket& packet : packets)
            {
                if (packet.Vertices.empty() || packet.Indices.empty() || !EnsureBuffers(packet.Vertices.size(), packet.Indices.size()))
                {
                    continue;
                }
                D3D11_MAPPED_SUBRESOURCE vertexMap{};
                D3D11_MAPPED_SUBRESOURCE indexMap{};
                if (FAILED(Context->Map(VertexBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &vertexMap)))
                {
                    continue;
                }
                if (FAILED(Context->Map(IndexBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &indexMap)))
                {
                    Context->Unmap(VertexBuffer.Get(), 0);
                    continue;
                }
                auto* vertices = static_cast<GPUVertex*>(vertexMap.pData);
                for (std::size_t index = 0; index < packet.Vertices.size(); ++index)
                {
                    const Vertex& source = packet.Vertices[index];
                    const Vector2 clip = interfaceSpace ? ProjectInterfaceToClip(source.Position, frame)
                                                        : ProjectWorldToClip(source.Position, frame);
                    vertices[index] = {clip.X, clip.Y, source.Tint.R, source.Tint.G, source.Tint.B, source.Tint.A, source.TexCoord.X, source.TexCoord.Y};
                }
                std::copy(packet.Indices.begin(), packet.Indices.end(), static_cast<std::uint32_t*>(indexMap.pData));
                Context->Unmap(VertexBuffer.Get(), 0);
                Context->Unmap(IndexBuffer.Get(), 0);

                const UINT stride = sizeof(GPUVertex);
                const UINT offset = 0;
                Context->IASetVertexBuffers(0, 1, VertexBuffer.GetAddressOf(), &stride, &offset);
                Context->IASetIndexBuffer(IndexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
                Context->IASetPrimitiveTopology(packet.Topology == PrimitiveTopology::Lines
                    ? D3D11_PRIMITIVE_TOPOLOGY_LINELIST : D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                if (packet.Field)
                {
                    if (!BindField(*packet.Field)) continue;
                }
                else Context->PSSetShader(PixelShader.Get(), nullptr, 0);
                Context->DrawIndexed(static_cast<UINT>(packet.Indices.size()), 0, 0);
            }
        }

        void DX11RHI::Draw(const FramePacket& frame)
        {
            if (!RenderTarget || frame.ViewportWidth <= 0 || frame.ViewportHeight <= 0)
            {
                return;
            }
            const float clear[4] = {frame.ClearColor.R, frame.ClearColor.G, frame.ClearColor.B, frame.ClearColor.A};
            Context->ClearRenderTargetView(RenderTarget.Get(), clear);
            Context->OMSetRenderTargets(1, RenderTarget.GetAddressOf(), nullptr);
            const D3D11_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(Width), static_cast<float>(Height), 0.0f, 1.0f};
            Context->RSSetViewports(1, &viewport);
            Context->IASetInputLayout(InputLayout.Get());
            Context->VSSetShader(VertexShader.Get(), nullptr, 0);
            Context->PSSetShader(PixelShader.Get(), nullptr, 0);
            Context->RSSetState(RasterizerState.Get());
            const float blendFactor[4] = {};
            Context->OMSetBlendState(BlendState.Get(), blendFactor, 0xffffffffu);
            DrawPackets(frame.WorldDraws, frame, false);
            DrawPackets(frame.UserInterfaceDraws, frame, true);
            if (!frame.CapturePath.empty()) Capture(frame.CapturePath);
            SwapChain->Present(1, 0);
        }

        void DX11RHI::Capture(const std::string& path)
        {
            ComPtr<ID3D11Texture2D> buffer;
            if (FAILED(SwapChain->GetBuffer(0, IID_PPV_ARGS(&buffer)))) return;
            D3D11_TEXTURE2D_DESC description{};
            buffer->GetDesc(&description);
            description.Usage = D3D11_USAGE_STAGING;
            description.BindFlags = 0;
            description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            ComPtr<ID3D11Texture2D> staging;
            if (FAILED(Device->CreateTexture2D(&description, nullptr, &staging))) return;
            Context->CopyResource(staging.Get(), buffer.Get());
            D3D11_MAPPED_SUBRESOURCE mapping{};
            if (FAILED(Context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapping))) return;
            std::ofstream output(path, std::ios::binary);
            output << "P6\n" << Width << " " << Height << "\n255\n";
            std::vector<char> row(static_cast<std::size_t>(Width * 3));
            for (int y = 0; y < Height; ++y)
            {
                const auto* source = static_cast<const char*>(mapping.pData) + y * mapping.RowPitch;
                for (int x = 0; x < Width; ++x) std::memcpy(row.data() + x * 3, source + x * 4, 3);
                output.write(row.data(), static_cast<std::streamsize>(row.size()));
            }
            Context->Unmap(staging.Get(), 0);
        }

        void DX11RHI::Shutdown()
        {
            if (Context) Context->ClearState();
            IndexBuffer.Reset(); VertexBuffer.Reset(); RasterizerState.Reset();
            BlendState.Reset(); InputLayout.Reset();
            FieldConstants.Reset(); FieldSampler.Reset(); FieldView.Reset(); FieldTexture.Reset();
            FieldPixelShader.Reset(); FieldWidth = FieldHeight = 0;
            PixelShader.Reset(); VertexShader.Reset(); RenderTarget.Reset(); SwapChain.Reset();
            Context.Reset(); Device.Reset();
        }
    }

    std::unique_ptr<IRHI> CreateDX11RHI()
    {
        return std::make_unique<DX11RHI>();
    }
}
