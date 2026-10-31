# CMake generated Testfile for 
# Source directory: F:/src/Riemann2d/Build
# Build directory: F:/src/Riemann2d/Build/Win64
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test([=[Riemann2DTests]=] "F:/src/Riemann2d/Binaries/Win64/Debug/Riemann2DTests.exe")
  set_tests_properties([=[Riemann2DTests]=] PROPERTIES  _BACKTRACE_TRIPLES "F:/src/Riemann2d/Build/CMakeLists.txt;238;add_test;F:/src/Riemann2d/Build/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test([=[Riemann2DTests]=] "F:/src/Riemann2d/Binaries/Win64/Release/Riemann2DTests.exe")
  set_tests_properties([=[Riemann2DTests]=] PROPERTIES  _BACKTRACE_TRIPLES "F:/src/Riemann2d/Build/CMakeLists.txt;238;add_test;F:/src/Riemann2d/Build/CMakeLists.txt;0;")
else()
  add_test([=[Riemann2DTests]=] NOT_AVAILABLE)
endif()
