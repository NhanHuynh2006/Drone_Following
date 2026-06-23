# Build: bộ điều khiển bay-theo PX4 (Jetson, C++ MAVSDK)
# ------------------------------------------------------
# Cần MAVSDK v2 cài sẵn trên Jetson (xem DEPLOY_GUIDE.md).
#   mkdir build && cd build && cmake .. && make -j4
#   ./follow_px4                       # bay thật (serial Pixhawk)
#   ./follow_px4 "udpin://0.0.0.0:14540"   # test SITL

cmake_minimum_required(VERSION 3.10)
project(follow_px4 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(MAVSDK REQUIRED)

add_executable(follow_px4 follow_px4.cpp)
target_link_libraries(follow_px4
    MAVSDK::mavsdk
    pthread
)
