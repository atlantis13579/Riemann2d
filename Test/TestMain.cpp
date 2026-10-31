#include "Test.h"

#include <iostream>

namespace Riemann2D::Test
{
    std::vector<Case>& Registry()
    {
        static std::vector<Case> cases;
        return cases;
    }

    Registrar::Registrar(const char* name, std::function<void()> function)
    {
        Registry().push_back({name, std::move(function)});
    }
}

int main(int argc, char** argv)
{
    int failureCount = 0;
    int runCount = 0;
    for (const Riemann2D::Test::Case& testCase : Riemann2D::Test::Registry())
    {
        if (argc > 1 && std::string(testCase.Name).find(argv[1]) == std::string::npos)
        {
            continue;
        }
        std::cout << "[RUN ] " << testCase.Name << '\n' << std::flush;
        ++runCount;
        try
        {
            testCase.Function();
            std::cout << "[PASS] " << testCase.Name << '\n' << std::flush;
        }
        catch (const std::exception& exception)
        {
            ++failureCount;
            std::cerr << "[FAIL] " << testCase.Name << ": " << exception.what() << '\n' << std::flush;
        }
    }

    if (runCount == 0)
    {
        std::cerr << "No tests matched the requested filter\n";
        return 2;
    }
    std::cout << runCount << " tests, "
              << failureCount << " failures\n";
    return failureCount == 0 ? 0 : 1;
}
