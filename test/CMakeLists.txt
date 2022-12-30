project(UnitTest)

### Unit Tests (TBD)
enable_testing()
include(GoogleTest)
#find_package(GTest CONFIG REQUIRED)
add_executable(unitTests)
target_include_directories(unitTests 
                          PUBLIC "${CMAKE_SOURCE_DIR}/include/"
                          PRIVATE ${LibArchive_INCLUDE_DIRS})
target_sources( unitTests
                PUBLIC
                  "TestRead.cpp"
                  "${CMAKE_SOURCE_DIR}/include/ExponentialAverage.h"
                  "${CMAKE_SOURCE_DIR}/src/ExponentialAverage.cpp"
                  "TestExponentialAverage.cpp")
target_link_libraries(unitTests 
                      PUBLIC ${OpenCV_LIBS} 
                      PRIVATE gtest gtest_main pthread
                      PUBLIC ${LibArchive_LIBRARIES})
gtest_discover_tests(unitTests)
file(COPY "testData" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")