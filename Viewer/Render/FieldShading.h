#pragma once

namespace Riemann2D::Render
{
    // Shared HLSL/Metal function body. The backend supplies SAMPLE_FIELD and lerp.
    inline constexpr char FieldShading[] = R"(
        float density = max(SAMPLE_FIELD(uv), 0.0);
        float2 texel = parameters.xy;
        float left = SAMPLE_FIELD(uv - float2(texel.x, 0.0));
        float right = SAMPLE_FIELD(uv + float2(texel.x, 0.0));
        float down = SAMPLE_FIELD(uv - float2(0.0, texel.y));
        float up = SAMPLE_FIELD(uv + float2(0.0, texel.y));
        float2 gradient = float2(right - left, up - down);
        if (parameters.z < 1.5) {
            float edgeWidth = max(fwidth(density), 0.035);
            float alpha = smoothstep(0.55 - edgeWidth, 0.55 + edgeWidth, density);
            float surface = 1.0 - smoothstep(0.8, 1.65, density);
            float sine = sin(parameters.w), cosine = cos(parameters.w);
            gradient = float2(cosine * gradient.x - sine * gradient.y,
                              sine * gradient.x + cosine * gradient.y);
            float3 normal = normalize(float3(-gradient * 2.2 * surface, 0.45));
            float3 light = normalize(float3(-0.45, 0.7, 0.65));
            float diffuse = max(dot(normal, light), 0.0);
            float specular = surface * pow(max(dot(normal, normalize(light + float3(0.0, 0.0, 1.0))), 0.0), 36.0);
            float thickness = saturate((density - 0.55) * 0.6);
            float3 color = lerp(float3(0.055, 0.43, 0.65), float3(0.018, 0.16, 0.32), thickness);
            color += float3(0.0, 0.12, 0.17) * saturate(uv.y);
            color *= 0.7 + 0.4 * diffuse;
            float rim = (1.0 - smoothstep(0.65, 1.35, density)) * saturate(-gradient.y * 3.0);
            color += float3(0.52, 0.78, 0.88) * (0.5 * specular + 0.55 * rim);
            return float4(color, alpha * (0.88 + 0.1 * thickness));
        }
        // Beer-Lambert extinction with short directional density taps for soft self-shadowing.
        float shadow = 0.0;
        shadow += SAMPLE_FIELD(uv + float2(-4.0, 6.0) * texel);
        shadow += SAMPLE_FIELD(uv + float2(-9.0, 13.0) * texel);
        shadow += SAMPLE_FIELD(uv + float2(-16.0, 23.0) * texel);
        float illumination = 0.34 + 0.66 * exp(-0.28 * shadow);
        float edgeLight = clamp(-gradient.x + gradient.y, -0.25, 0.25);
        float3 smokeColor = float3(0.65, 0.76, 0.86) * (illumination + 0.3 * edgeLight);
        return float4(smokeColor, 1.0 - exp(-1.6 * density));
    )";
}
