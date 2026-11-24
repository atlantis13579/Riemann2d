#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include "Viewer.h"

#include <chrono>
#include <thread>

namespace
{
    struct ApplicationState
    {
        Riemann2D::Viewer::Viewer Viewer;
        Riemann2D::Render::InterfaceInput Input;
        bool Running = true;
    };

    LRESULT CALLBACK WindowProcedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
    {
        auto* state = reinterpret_cast<ApplicationState*>(GetWindowLongPtrW(window, GWLP_USERDATA));
        switch (message)
        {
        case WM_NCCREATE:
        {
            const auto* create = reinterpret_cast<CREATESTRUCTW*>(lParam);
            SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
            return DefWindowProcW(window, message, wParam, lParam);
        }
        case WM_SIZE:
            if (state && wParam != SIZE_MINIMIZED)
            {
                state->Viewer.Resize(LOWORD(lParam), HIWORD(lParam));
            }
            return 0;
        case WM_MOUSEMOVE:
            if (state)
            {
                state->Input.MouseX = static_cast<short>(LOWORD(lParam));
                state->Input.MouseY = static_cast<short>(HIWORD(lParam));
            }
            return 0;
        case WM_LBUTTONDOWN:
            if (state)
            {
                state->Input.MouseDown = true;
                state->Input.MousePressed = true;
                SetCapture(window);
            }
            return 0;
        case WM_LBUTTONUP:
            if (state)
            {
                state->Input.MouseDown = false;
                ReleaseCapture();
            }
            return 0;
        case WM_DESTROY:
            if (state) state->Running = false;
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(window, message, wParam, lParam);
        }
    }
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR commandLine, int showCommand)
{
    const wchar_t className[] = L"Riemann2DViewerWindow";
    WNDCLASSEXW windowClass{};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = WindowProcedure;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
    windowClass.lpszClassName = className;
    if (!RegisterClassExW(&windowClass)) return 1;

    ApplicationState state;
    HWND window = CreateWindowExW(0, className, L"Riemann2D - DX11",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1280, 760,
        nullptr, nullptr, instance, &state);
    if (!window) return 2;
    ShowWindow(window, showCommand);

    RECT client{};
    GetClientRect(window, &client);
    if (!state.Viewer.Initialize(window, client.right - client.left, client.bottom - client.top)) return 3;
    if (std::wstring(commandLine).find(L"snow_simulation_2d") != std::wstring::npos)
        state.Viewer.SelectScene("snow_simulation_2d");
    else if (std::wstring(commandLine).find(L"domino_chain_2d") != std::wstring::npos)
        state.Viewer.SelectScene("domino_chain_2d");

    auto previous = std::chrono::steady_clock::now();
    MSG message{};
    while (state.Running)
    {
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
        {
            if (message.message == WM_QUIT) state.Running = false;
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        const auto current = std::chrono::steady_clock::now();
        const float deltaTime = std::chrono::duration<float>(current - previous).count();
        previous = current;
        state.Viewer.Tick(deltaTime, state.Input);
        state.Input.MousePressed = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }

    state.Viewer.Shutdown();
    return 0;
}
