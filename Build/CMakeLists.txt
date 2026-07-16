cmake_minimum_required(VERSION 3.20)
project(Riemann2D VERSION 0.1.0 LANGUAGES CXX)

option(RIEMANN2D_BUILD_VIEWER "Build the interactive viewer" ON)
option(RIEMANN2D_BUILD_TESTS "Build the test executable" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE STRING "Supported configurations" FORCE)
endif()

get_filename_component(RIEMANN2D_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

if(WIN32)
    set(RIEMANN2D_PLATFORM_NAME Win64)
elseif(APPLE)
    set(RIEMANN2D_PLATFORM_NAME Mac)
else()
    set(RIEMANN2D_PLATFORM_NAME Linux)
endif()

set(RIEMANN2D_OUTPUT_DIRECTORY "${RIEMANN2D_ROOT_DIR}/Binaries/${RIEMANN2D_PLATFORM_NAME}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${RIEMANN2D_OUTPUT_DIRECTORY}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${RIEMANN2D_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${RIEMANN2D_OUTPUT_DIRECTORY}")

foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${RIEMANN2D_OUTPUT_DIRECTORY}/${config}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${RIEMANN2D_OUTPUT_DIRECTORY}/${config}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${RIEMANN2D_OUTPUT_DIRECTORY}/${config}")
endforeach()

if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Threads REQUIRED)

file(GLOB_RECURSE RIEMANN2D_SOURCES CONFIGURE_DEPENDS
    "${RIEMANN2D_ROOT_DIR}/Src/*.cpp"
    "${RIEMANN2D_ROOT_DIR}/Src/*.cc"
    "${RIEMANN2D_ROOT_DIR}/Src/*.c"
)

file(GLOB_RECURSE RIEMANN2D_HEADERS CONFIGURE_DEPENDS
    "${RIEMANN2D_ROOT_DIR}/Src/*.h"
    "${RIEMANN2D_ROOT_DIR}/Src/*.hpp"
)

add_library(Riemann2D STATIC ${RIEMANN2D_SOURCES} ${RIEMANN2D_HEADERS})
target_include_directories(Riemann2D PUBLIC "${RIEMANN2D_ROOT_DIR}/Src")
target_compile_features(Riemann2D PUBLIC cxx_std_17)

set(RIEMANN2D_VIEWER_RENDER_SOURCES
    "${RIEMANN2D_ROOT_DIR}/Viewer/Render/ImmediateUI.cpp"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Render/ImmediateUI.h"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Render/Projection.h"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RenderThread.cpp"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RenderThread.h"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RenderTypes.h"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Render/Renderer2.cpp"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Render/Renderer2.h"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/RHI.h"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/Null/NullRHI.cpp"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/Null/NullRHI.h"
)

set(RIEMANN2D_VIEWER_COMMON_SOURCES
    "${RIEMANN2D_ROOT_DIR}/Viewer/FluidSurfaceBuilder.cpp"
    "${RIEMANN2D_ROOT_DIR}/Viewer/FluidSurfaceBuilder.h"
    "${RIEMANN2D_ROOT_DIR}/Viewer/GeometryBuilder.cpp"
    "${RIEMANN2D_ROOT_DIR}/Viewer/GeometryBuilder.h"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Viewer.cpp"
    "${RIEMANN2D_ROOT_DIR}/Viewer/Viewer.h"
    ${RIEMANN2D_VIEWER_RENDER_SOURCES}
)

set(RIEMANN2D_VIEWER_PLATFORM_SOURCES)
if(WIN32)
    list(APPEND RIEMANN2D_VIEWER_PLATFORM_SOURCES
        "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/DX11/DX11RHI.cpp"
        "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/DX11/DX11RHI.h"
        "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/PlatformRHI.cpp"
        "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/PlatformRHI.h"
    )
elseif(APPLE)
    enable_language(OBJCXX)
    list(APPEND RIEMANN2D_VIEWER_PLATFORM_SOURCES
        "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/Metal/MetalRHI.mm"
        "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/Metal/MetalRHI.h"
        "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/PlatformRHI.cpp"
        "${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/PlatformRHI.h"
    )
    set_source_files_properties("${RIEMANN2D_ROOT_DIR}/Viewer/Render/RHI/Metal/MetalRHI.mm"
        PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

if(RIEMANN2D_BUILD_VIEWER AND (WIN32 OR APPLE))
    if(WIN32)
        add_executable(Viewer WIN32
            ${RIEMANN2D_VIEWER_COMMON_SOURCES}
            ${RIEMANN2D_VIEWER_PLATFORM_SOURCES}
            "${RIEMANN2D_ROOT_DIR}/Viewer/WinApplication.cpp"
        )
        target_link_libraries(Viewer PRIVATE
            Riemann2D
            Threads::Threads
            user32
            d3d11
            d3dcompiler
            dxgi
        )
        set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT Viewer)
    elseif(APPLE)
        add_executable(Viewer MACOSX_BUNDLE
            ${RIEMANN2D_VIEWER_COMMON_SOURCES}
            ${RIEMANN2D_VIEWER_PLATFORM_SOURCES}
            "${RIEMANN2D_ROOT_DIR}/Viewer/MacApplication.mm"
        )
        set_source_files_properties("${RIEMANN2D_ROOT_DIR}/Viewer/MacApplication.mm"
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
        find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
        find_library(METAL_FRAMEWORK Metal REQUIRED)
        find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
        target_link_libraries(Viewer PRIVATE
            Riemann2D
            Threads::Threads
            ${COCOA_FRAMEWORK}
            ${METAL_FRAMEWORK}
            ${QUARTZCORE_FRAMEWORK}
        )
    endif()

    target_include_directories(Viewer PRIVATE
        "${RIEMANN2D_ROOT_DIR}/Viewer"
        "${RIEMANN2D_ROOT_DIR}/Viewer/Render"
        "${RIEMANN2D_ROOT_DIR}/Src"
    )
    target_compile_features(Viewer PUBLIC cxx_std_17)
endif()

if(RIEMANN2D_BUILD_TESTS)
    enable_testing()
    file(GLOB RIEMANN2D_TEST_SOURCES CONFIGURE_DEPENDS
        "${RIEMANN2D_ROOT_DIR}/Test/*.cpp"
        "${RIEMANN2D_ROOT_DIR}/Test/*.cc"
        "${RIEMANN2D_ROOT_DIR}/Test/*.c"
    )

    add_executable(Riemann2DTests
        ${RIEMANN2D_TEST_SOURCES}
        ${RIEMANN2D_VIEWER_RENDER_SOURCES}
    )
    target_include_directories(Riemann2DTests PRIVATE
        "${RIEMANN2D_ROOT_DIR}/Test"
        "${RIEMANN2D_ROOT_DIR}/Viewer"
        "${RIEMANN2D_ROOT_DIR}/Viewer/Render"
        "${RIEMANN2D_ROOT_DIR}/Src"
    )
    target_link_libraries(Riemann2DTests PRIVATE Riemann2D Threads::Threads)
    add_test(NAME Riemann2DTests COMMAND Riemann2DTests)
endif()

source_group(TREE "${RIEMANN2D_ROOT_DIR}/Src" PREFIX "Src"
    FILES ${RIEMANN2D_SOURCES} ${RIEMANN2D_HEADERS})

source_group(TREE "${RIEMANN2D_ROOT_DIR}/Viewer" PREFIX "Viewer"
    FILES ${RIEMANN2D_VIEWER_COMMON_SOURCES} ${RIEMANN2D_VIEWER_PLATFORM_SOURCES})

if(DEFINED RIEMANN2D_TEST_SOURCES)
    source_group(TREE "${RIEMANN2D_ROOT_DIR}/Test" PREFIX "Test"
        FILES ${RIEMANN2D_TEST_SOURCES})
endif()
