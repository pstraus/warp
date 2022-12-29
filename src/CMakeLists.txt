cmake_minimum_required( VERSION 3.16.3 )
add_executable(main "main.cpp")

include(OpenCV REQUIRED)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})