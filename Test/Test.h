#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Riemann2D::Test
{
    struct Case
    {
        const char* Name;
        std::function<void()> Function;
    };

    std::vector<Case>& Registry();

    struct Registrar
    {
        Registrar(const char* name, std::function<void()> function);
    };

    inline void Require(bool condition, const char* expression, const char* file, int line)
    {
        if (!condition)
        {
            throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                     " requirement failed: " + expression);
        }
    }
}

#define R2_JOIN_IMPL(a, b) a##b
#define R2_JOIN(a, b) R2_JOIN_IMPL(a, b)
#define R2_TEST(name) \
    static void R2_JOIN(R2TestFunction_, __LINE__)(); \
    static ::Riemann2D::Test::Registrar R2_JOIN(R2TestRegistrar_, __LINE__)( \
        name, R2_JOIN(R2TestFunction_, __LINE__)); \
    static void R2_JOIN(R2TestFunction_, __LINE__)()
#define R2_REQUIRE(expression) \
    ::Riemann2D::Test::Require(static_cast<bool>(expression), #expression, __FILE__, __LINE__)
