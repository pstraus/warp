### Include VCPKG
#set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")

cmake_minimum_required(VERSION 3.24)

#Project name
project(warp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)
#set(CMAKE_CXX_FLAGS "-Wl, -Wno-undef")
#set(CMAKE_CXX_FLAGS -lstdc++)

#VCPKG compiled opencv using gcc...
#if(CMAKE_CXX_COMPILER MATCHES clang)
#set(CMAKE_CXX_FLAGS -stdlib="libstdc++")
#endif()

add_subdirectory(src)

find_package(LibArchive REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(FFMPEG REQUIRED)

add_executable(main)
target_sources( main 
                PRIVATE "src/main.cpp"
                PRIVATE "src/ExponentialAverage.cpp"
                PRIVATE "src/WarpingProcessor.cpp"
                PRIVATE "src/"
              )
target_include_directories(main 
                          PUBLIC "${CMAKE_SOURCE_DIR}/include" 
                          PUBLIC ${LibArchive_INCLUDE_DIRS}
			  PRIVATE ${FFMPEG_INCLUDE_DIRS}
)
target_link_libraries(main 
                      PUBLIC ${OpenCV_LIBS} 
                      PUBLIC ${LibArchive_LIBRARIES}
		      PRIVATE ${FFMPEG_LIBRARIES}
)
target_link_directories(main
	PRIVATE ${FFMPEG_LIBRARY_DIRS}
)
### Unit Tests (TBD)

add_subdirectory(test)
file(COPY "testData" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
#enable_testing()
#include(GoogleTest)
#find_package(GTest CONFIG REQUIRED)
#add_executable(unitTests)
#target_include_directories(unitTests PRIVATE "${CMAKE_SOURCE_DIR}/include")
#target_sources( unitTests
#                PRIVATE
#                  "test/TestRead.cpp"
#                  "include/ExponentialAverage.h"
#                  "test/TestExponentialAverage.cpp")
#target_link_libraries(unitTests PRIVATE ${OpenCV_LIBS} gtest gtest_main pthread)
#gtest_discover_tests(unitTests)
