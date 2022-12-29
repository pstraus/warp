cmake_minimum_required( VERSION 3.16.3 )
#file(TO_CMAKE_PATH $ENV{LOCAL_ROOT} LOCAL_ROOT)
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")

#Project name
project(warp)
#include (cmake/dist.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)

include_directories(include)
add_subdirectory(src)
add_subdirectory(test)

find_package(OpenCV REQUIRED)

#add_executable(Test "main.cpp")
add_executable(main)
target_sources( main 
                PRIVATE
                "src/main.cpp"
              )
target_link_libraries(main ${OpenCV_LIBS})

### Unit Tests (TBD)
enable_testing()
include(GoogleTest)
find_package(GTest CONFIG REQUIRED)
add_executable(unitTests)
target_sources( unitTests
                PRIVATE
                "test/TestRead.cpp")
target_link_libraries(unitTests PRIVATE ${OpenCV_LIBS} gtest gtest_main pthread)
gtest_discover_tests(unitTests)
