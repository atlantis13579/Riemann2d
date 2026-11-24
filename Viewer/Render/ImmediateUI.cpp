#include "ImmediateUI.h"

#include "Maths/Maths.h"

#include <array>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <utility>

namespace Riemann2D::Render
{
    namespace
    {
        struct Glyph
        {
            char Character;
            std::array<std::uint8_t, 5> Columns;
        };

        // This compact font is deliberately limited to labels used by the viewer.
        constexpr Glyph Font[] = {
            {' ', {0,0,0,0,0}}, {'-', {8,8,8,8,8}}, {'.', {0,0,0,24,24}}, {':', {0,10,0,10,0}},
            {'0', {30,33,33,33,30}}, {'1', {0,34,63,32,0}}, {'2', {50,41,41,41,38}},
            {'3', {18,33,37,37,26}}, {'4', {12,10,9,63,8}}, {'5', {23,37,37,37,25}},
            {'6', {30,37,37,37,24}}, {'7', {1,1,57,5,3}}, {'8', {26,37,37,37,26}},
            {'9', {6,41,41,41,30}},
            {'A', {62,9,9,9,62}}, {'B', {63,37,37,37,26}}, {'C', {30,33,33,33,18}},
            {'D', {63,33,33,18,12}}, {'E', {63,37,37,37,33}}, {'F', {63,5,5,5,1}},
            {'G', {30,33,41,41,58}}, {'H', {63,4,4,4,63}}, {'I', {33,33,63,33,33}},
            {'J', {16,32,32,32,31}}, {'K', {63,4,10,17,32}}, {'L', {63,32,32,32,32}},
            {'M', {63,2,4,2,63}}, {'N', {63,2,4,8,63}}, {'O', {30,33,33,33,30}},
            {'P', {63,9,9,9,6}}, {'Q', {30,33,41,17,46}}, {'R', {63,9,25,41,38}},
            {'S', {38,37,37,37,25}}, {'T', {1,1,63,1,1}}, {'U', {31,32,32,32,31}},
            {'V', {15,16,32,16,15}}, {'W', {63,16,8,16,63}}, {'X', {33,18,12,18,33}},
            {'Y', {3,4,56,4,3}}, {'Z', {49,41,37,35,33}}
        };

        const std::array<std::uint8_t, 5>& FindGlyph(char character)
        {
            static constexpr std::array<std::uint8_t, 5> unknown{33,18,12,0,12};
            const char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
            for (const Glyph& glyph : Font)
            {
                if (glyph.Character == upper)
                {
                    return glyph.Columns;
                }
            }
            return unknown;
        }
    }

    void ImmediateUI::Begin(int width, int height, int panelWidth, const InterfaceInput& input)
    {
        (void)width;
        Draw = {};
        Draw.DebugName = "ImmediateUI";
        Draw.Topology = PrimitiveTopology::Triangles;
        Input = input;
        PanelWidth = panelWidth;
        CursorY = 18.0f;
        AddRectangle(0.0f, 0.0f, static_cast<float>(panelWidth), static_cast<float>(height),
                     {0.055f, 0.065f, 0.09f, 1.0f});
        AddRectangle(static_cast<float>(panelWidth - 1), 0.0f, static_cast<float>(panelWidth),
                     static_cast<float>(height), {0.2f, 0.24f, 0.32f, 1.0f});
    }

    void ImmediateUI::Label(const std::string& text, const Color& color)
    {
        const float scale = std::min(2.0f, (static_cast<float>(PanelWidth) - 28.0f) /
            (6.0f * static_cast<float>(std::max<std::size_t>(text.size(), 1))));
        AddText(14.0f, CursorY, text, color, scale);
        CursorY += 20.0f;
    }

    bool ImmediateUI::Button(const std::string& text, bool selected, bool enabled)
    {
        const float left = 12.0f;
        const float top = CursorY;
        const float right = static_cast<float>(PanelWidth - 12);
        const float bottom = top + 28.0f;
        const bool hovered = enabled && IsHovered(left, top, right, bottom);
        Color color = selected ? Color{0.15f, 0.42f, 0.7f, 1.0f} : Color{0.11f, 0.13f, 0.18f, 1.0f};
        if (hovered)
        {
            color = selected ? Color{0.2f, 0.5f, 0.8f, 1.0f} : Color{0.18f, 0.21f, 0.29f, 1.0f};
        }
        AddRectangle(left, top, right, bottom, color);
        const float scale = std::min(2.0f, (right - left - 18.0f) /
            (6.0f * static_cast<float>(std::max<std::size_t>(text.size(), 1))));
        AddText(left + 9.0f, top + 7.0f, text, enabled ? Color{0.92f, 0.94f, 0.98f, 1.0f}
                                                     : Color{0.38f, 0.42f, 0.48f, 1.0f}, scale);
        CursorY = bottom + 6.0f;
        return hovered && Input.MousePressed;
    }

    void ImmediateUI::Separator()
    {
        const float y = CursorY + 7.0f;
        AddRectangle(12.0f, y, static_cast<float>(PanelWidth - 12), y + 1.0f,
                     {0.2f, 0.24f, 0.32f, 1.0f});
        CursorY += 20.0f;
    }

    bool ImmediateUI::Slider(const std::string& text, float& value, float minimum, float maximum)
    {
        Label(text);
        const float left = 14.0f;
        const float top = CursorY;
        const float right = static_cast<float>(PanelWidth - 14);
        const float bottom = top + 18.0f;
        bool changed = false;
        if (IsHovered(left, top, right, bottom) && Input.MouseDown)
        {
            const float fraction = Clamp((static_cast<float>(Input.MouseX) - left) / (right - left), 0.0f, 1.0f);
            value = minimum + fraction * (maximum - minimum);
            changed = true;
        }
        const float fraction = Clamp((value - minimum) / (maximum - minimum), 0.0f, 1.0f);
        AddRectangle(left, top + 6.0f, right, top + 12.0f, {0.12f, 0.14f, 0.19f, 1.0f});
        AddRectangle(left, top + 6.0f, left + fraction * (right - left), top + 12.0f,
                     {0.22f, 0.62f, 0.85f, 1.0f});
        AddRectangle(left + fraction * (right - left) - 3.0f, top + 2.0f,
                     left + fraction * (right - left) + 3.0f, bottom - 2.0f,
                     {0.8f, 0.9f, 1.0f, 1.0f});
        CursorY = bottom + 10.0f;
        return changed;
    }

    DrawPacket ImmediateUI::End()
    {
        return std::move(Draw);
    }

    void ImmediateUI::AddRectangle(float left, float top, float right, float bottom, const Color& color)
    {
        const std::uint32_t first = static_cast<std::uint32_t>(Draw.Vertices.size());
        Draw.Vertices.push_back({{left, top}, color});
        Draw.Vertices.push_back({{right, top}, color});
        Draw.Vertices.push_back({{right, bottom}, color});
        Draw.Vertices.push_back({{left, bottom}, color});
        Draw.Indices.insert(Draw.Indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
    }

    void ImmediateUI::AddText(float x, float y, const std::string& text, const Color& color, float scale)
    {
        for (char character : text)
        {
            const auto& columns = FindGlyph(character);
            for (int column = 0; column < 5; ++column)
            {
                for (int row = 0; row < 6; ++row)
                {
                    if ((columns[static_cast<std::size_t>(column)] & (1u << row)) != 0)
                    {
                        const float left = x + static_cast<float>(column) * scale;
                        const float top = y + static_cast<float>(row) * scale;
                        AddRectangle(left, top, left + scale, top + scale, color);
                    }
                }
            }
            x += 6.0f * scale;
        }
    }

    bool ImmediateUI::IsHovered(float left, float top, float right, float bottom) const
    {
        return static_cast<float>(Input.MouseX) >= left && static_cast<float>(Input.MouseX) <= right &&
               static_cast<float>(Input.MouseY) >= top && static_cast<float>(Input.MouseY) <= bottom;
    }
}
