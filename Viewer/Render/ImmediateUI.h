#pragma once

#include "RenderTypes.h"

#include <string>

namespace Riemann2D::Render
{
    struct InterfaceInput
    {
        int MouseX = 0;
        int MouseY = 0;
        bool MouseDown = false;
        bool MousePressed = false;
    };

    class ImmediateUI
    {
    public:
        void Begin(int width, int height, int panelWidth, const InterfaceInput& input,
                   float displayScale = 1.0f);
        void Label(const std::string& text, const Color& color = {0.85f, 0.88f, 0.95f, 1.0f});
        bool Button(const std::string& text, bool selected = false, bool enabled = true);
        void Separator();
        bool Slider(const std::string& text, float& value, float minimum, float maximum);
        DrawPacket End();

    private:
        void AddRectangle(float left, float top, float right, float bottom, const Color& color);
        void AddText(float x, float y, const std::string& text, const Color& color, float scale = 2.0f);
        bool IsHovered(float left, float top, float right, float bottom) const;

        DrawPacket Draw;
        InterfaceInput Input;
        float CursorY = 0.0f;
        float PanelWidth = 0.0f;
        float DisplayScale = 1.0f;
    };
}
