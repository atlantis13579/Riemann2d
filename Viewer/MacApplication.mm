#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

#include "Viewer.h"

#include <chrono>
#include <memory>

@interface Riemann2DAppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@end

@implementation Riemann2DAppDelegate
{
    NSWindow* Window;
    CAMetalLayer* MetalLayer;
    NSTimer* Timer;
    std::unique_ptr<Riemann2D::Viewer::Viewer> Viewer;
    std::chrono::steady_clock::time_point PreviousTime;
    bool PreviousMouseDown;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification
{
    (void)notification;
    const NSRect frame = NSMakeRect(120.0, 120.0, 1280.0, 760.0);
    Window = [[NSWindow alloc] initWithContentRect:frame
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
        backing:NSBackingStoreBuffered defer:NO];
    Window.title = @"Riemann2D - Metal";
    Window.delegate = self;
    NSView* view = Window.contentView;
    view.wantsLayer = YES;
    MetalLayer = [CAMetalLayer layer];
    view.layer = MetalLayer;
    const NSSize size = view.bounds.size;
    const CGFloat scale = Window.backingScaleFactor;
    MetalLayer.contentsScale = scale;
    MetalLayer.drawableSize = CGSizeMake(size.width * scale, size.height * scale);

    Viewer = std::make_unique<Riemann2D::Viewer::Viewer>();
    Viewer->SetDisplayScale(static_cast<float>(scale));
    if (!Viewer->Initialize((__bridge void*)MetalLayer,
                            static_cast<int>(size.width * scale), static_cast<int>(size.height * scale)))
    {
        [NSApp terminate:nil];
        return;
    }
    PreviousTime = std::chrono::steady_clock::now();
    if ([NSProcessInfo.processInfo.arguments containsObject:@"snow_simulation_2d"])
        Viewer->SelectScene("snow_simulation_2d");
    else if ([NSProcessInfo.processInfo.arguments containsObject:@"domino_chain_2d"])
        Viewer->SelectScene("domino_chain_2d");
    PreviousMouseDown = false;
    Timer = [NSTimer scheduledTimerWithTimeInterval:1.0 / 120.0 target:self
        selector:@selector(tick:) userInfo:nil repeats:YES];
    [Window makeKeyAndOrderFront:nil];
}

- (void)tick:(NSTimer*)timer
{
    (void)timer;
    const auto current = std::chrono::steady_clock::now();
    const float deltaTime = std::chrono::duration<float>(current - PreviousTime).count();
    PreviousTime = current;

    NSView* view = Window.contentView;
    const CGFloat scale = Window.backingScaleFactor;
    MetalLayer.contentsScale = scale;
    const NSSize size = view.bounds.size;
    const int width = static_cast<int>(size.width * scale);
    const int height = static_cast<int>(size.height * scale);
    MetalLayer.drawableSize = CGSizeMake(width, height);
    Viewer->SetDisplayScale(static_cast<float>(scale));
    Viewer->Resize(width, height);

    NSPoint mouse = [view convertPoint:Window.mouseLocationOutsideOfEventStream fromView:nil];
    const bool mouseDown = (NSEvent.pressedMouseButtons & 1u) != 0;
    Riemann2D::Render::InterfaceInput input;
    input.MouseX = static_cast<int>(mouse.x * scale);
    input.MouseY = height - static_cast<int>(mouse.y * scale);
    input.MouseDown = mouseDown;
    input.MousePressed = mouseDown && !PreviousMouseDown;
    PreviousMouseDown = mouseDown;
    Viewer->Tick(deltaTime, input);
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender
{
    (void)sender;
    return YES;
}

- (void)applicationWillTerminate:(NSNotification*)notification
{
    (void)notification;
    [Timer invalidate];
    if (Viewer)
    {
        Viewer->Shutdown();
        Viewer.reset();
    }
}
@end

int main(int argc, const char* argv[])
{
    (void)argc;
    (void)argv;
    @autoreleasepool
    {
        NSApplication* application = [NSApplication sharedApplication];
        Riemann2DAppDelegate* delegate = [[Riemann2DAppDelegate alloc] init];
        application.delegate = delegate;
        [application run];
    }
    return 0;
}
