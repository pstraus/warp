cmake_minimum_required( VERSION 3.16.3 )
#file(TO_CMAKE_PATH $ENV{LOCAL_ROOT} LOCAL_ROOT)
set(CMAKE_TOOLCHAIN_FILE "${PROJECT_SOURCE_DIR}vcpkg/scripts/buildsystems/vcpkg.cmake")

#Project name
project(warp)
#include (cmake/dist.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)


include(GoogleTest)
find_package(OpenCV REQUIRED)
#find_package(GTest CONFIG REQUIRED)

#add_executable(Test "main.cpp")
add_executable(main "main.cpp")

target_link_libraries(main ${OpenCV_LIBS})

### Unit Tests (TBD)
#target_link_libraries(Test PRIVATE ${OpenCV_LIBS})

#gtest_discover_tests(Test)
