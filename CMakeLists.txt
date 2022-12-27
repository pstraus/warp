cmake_minimum_required( VERSION 3.16.3 )
project(warp)
#include (cmake/dist.cmake)

#set OpenCV_DIR; let's intellisense find stuff immediately
#set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-linux/share/opencv4")

find_package(OpenCV REQUIRED)
#include_directories(${})
