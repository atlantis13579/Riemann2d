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

set(RIEMANN2D_OUTPUT_DIRECTORY "${RIEMANN2D_ROOT_DIR}/Binaries/${RIEMANN2D_PLATFORM_NAME}"
    CACHE PATH "Build output directory")
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
    "${RIEMANN2D_ROOT_DIR}/src/*.cpp"
    "${RIEMANN2D_ROOT_DIR}/src/*.cc"
    "${RIEMANN2D_ROOT_DIR}/src/*.c"
)

file(GLOB_RECURSE RIEMANN2D_HEADERS CONFIGURE_DEPENDS
    "${RIEMANN2D_ROOT_DIR}/src/*.h"
    "${RIEMANN2D_ROOT_DIR}/src/*.hpp"
)

add_library(Riemann2D STATIC ${RIEMANN2D_SOURCES} ${RIEMANN2D_HEADERS})
target_include_directories(Riemann2D PUBLIC "${RIEMANN2D_ROOT_DIR}/src")
target_compile_features(Riemann2D PUBLIC cxx_std_17)

option(RIEMANN2D_ENABLE_COMPUTE_SHADER "Build native DX11/Metal compute fluid solvers" ON)
if(RIEMANN2D_ENABLE_COMPUTE_SHADER)
    if(WIN32)
        target_compile_definitions(Riemann2D PRIVATE RIEMANN2D_HAS_COMPUTE_SHADER=1)
        target_link_libraries(Riemann2D PRIVATE d3d11 d3dcompiler dxgi)
    elseif(APPLE)
        enable_language(OBJCXX)
        target_sources(Riemann2D PRIVATE "${RIEMANN2D_ROOT_DIR}/src/Fluid/Compute/MetalComputeDevice.mm")
        set_property(TARGET Riemann2D PROPERTY OBJCXX_STANDARD 17)
        set_source_files_properties("${RIEMANN2D_ROOT_DIR}/src/Fluid/Compute/MetalComputeDevice.mm"
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
        target_compile_definitions(Riemann2D PRIVATE RIEMANN2D_HAS_COMPUTE_SHADER=1)
        find_library(COMPUTE_METAL_FRAMEWORK Metal REQUIRED)
        find_library(COMPUTE_FOUNDATION_FRAMEWORK Foundation REQUIRED)
        target_link_libraries(Riemann2D PRIVATE ${COMPUTE_METAL_FRAMEWORK} ${COMPUTE_FOUNDATION_FRAMEWORK})
    endif()
endif()

option(RIEMANN2D_ENABLE_CUDA "Build optional CUDA fluid solvers when a toolkit is available" ON)
set(RIEMANN2D_CUDA_ARCHITECTURES "75;86" CACHE STRING "CUDA real and virtual architectures")
if(RIEMANN2D_ENABLE_CUDA AND NOT APPLE)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_NVCC_EXECUTABLE)
        # Compile with nvcc directly so standalone toolkits do not need VS build customizations.
        set(cuda_architecture_flags)
        foreach(architecture IN LISTS RIEMANN2D_CUDA_ARCHITECTURES)
            list(APPEND cuda_architecture_flags
                "--generate-code=arch=compute_${architecture},code=[sm_${architecture},compute_${architecture}]")
        endforeach()
        set(cuda_host_flags)
        if(MSVC)
            list(APPEND cuda_host_flags "-Xcompiler=$<IF:$<CONFIG:Debug>,/MDd,/MD>" "-Xcompiler=/EHsc")
        else()
            list(APPEND cuda_host_flags "-Xcompiler=-fPIC")
        endif()
        foreach(cuda_path Fluid/Cuda/CudaFluidSolver Fluid/Cuda/CudaAirSolver Continuum/CudaSnowSolver2D)
            get_filename_component(cuda_source "${cuda_path}" NAME)
            set(cuda_object "${CMAKE_CURRENT_BINARY_DIR}/Cuda/$<CONFIG>/${cuda_source}${CMAKE_CXX_OUTPUT_EXTENSION}")
            add_custom_command(OUTPUT "${cuda_object}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/Cuda/$<CONFIG>"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++17 -O3 --fmad=false
                    ${cuda_architecture_flags} ${cuda_host_flags} -ccbin "${CMAKE_CXX_COMPILER}"
                    "-I${RIEMANN2D_ROOT_DIR}/src" -c "${RIEMANN2D_ROOT_DIR}/src/${cuda_path}.cu"
                    -o "${cuda_object}"
                DEPENDS "${RIEMANN2D_ROOT_DIR}/src/${cuda_path}.cu"
                    ${RIEMANN2D_HEADERS}
                    "${RIEMANN2D_ROOT_DIR}/src/Fluid/Cuda/CudaCommon.cuh"
                    "${RIEMANN2D_ROOT_DIR}/src/Fluid/Cuda/CudaSolvers.h"
                    "${RIEMANN2D_ROOT_DIR}/src/Fluid/FluidSimulation2D.h"
                    "${RIEMANN2D_ROOT_DIR}/src/Fluid/AerodynamicsSimulation2D.h"
                VERBATIM)
            set_source_files_properties("${cuda_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
            target_sources(Riemann2D PRIVATE "${cuda_object}")
        endforeach()
        target_compile_definitions(Riemann2D PRIVATE RIEMANN2D_HAS_CUDA=1)
        target_link_libraries(Riemann2D PRIVATE CUDA::cudart_static)
        message(STATUS "Riemann2D CUDA solvers enabled (${CUDAToolkit_VERSION})")
    else()
        message(STATUS "CUDA compiler/toolkit not found; building CPU fluid solvers")
    endif()
endif()

option(RIEMANN2D_PARALLEL_FLUIDS "Parallelize large fluid grids when OpenMP is available" ON)
if(RIEMANN2D_PARALLEL_FLUIDS)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(Riemann2D PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

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
    "${RIEMANN2D_ROOT_DIR}/Viewer/DominoChainDemo.cpp"
    "${RIEMANN2D_ROOT_DIR}/Viewer/DominoChainDemo.h"
    "${RIEMANN2D_ROOT_DIR}/Viewer/SnowSurfaceBuilder.cpp"
    "${RIEMANN2D_ROOT_DIR}/Viewer/SnowSurfaceBuilder.h"
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
        "${RIEMANN2D_ROOT_DIR}/src"
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
        "${RIEMANN2D_ROOT_DIR}/Viewer/FluidSurfaceBuilder.cpp"
    )
    target_include_directories(Riemann2DTests PRIVATE
        "${RIEMANN2D_ROOT_DIR}/Test"
        "${RIEMANN2D_ROOT_DIR}/Viewer"
        "${RIEMANN2D_ROOT_DIR}/Viewer/Render"
        "${RIEMANN2D_ROOT_DIR}/src"
    )
    target_link_libraries(Riemann2DTests PRIVATE Riemann2D Threads::Threads)
    add_test(NAME Riemann2DTests COMMAND Riemann2DTests)
endif()

source_group(TREE "${RIEMANN2D_ROOT_DIR}/src" PREFIX "Src"
    FILES ${RIEMANN2D_SOURCES} ${RIEMANN2D_HEADERS})

source_group(TREE "${RIEMANN2D_ROOT_DIR}/Viewer" PREFIX "Viewer"
    FILES ${RIEMANN2D_VIEWER_COMMON_SOURCES} ${RIEMANN2D_VIEWER_PLATFORM_SOURCES})

if(DEFINED RIEMANN2D_TEST_SOURCES)
    source_group(TREE "${RIEMANN2D_ROOT_DIR}/Test" PREFIX "Test"
        FILES ${RIEMANN2D_TEST_SOURCES})
endif()
