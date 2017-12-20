cmake_minimum_required(VERSION 3.5)
project(GPUFlow C CXX)

set(CMAKE_CXX_STANDARD 11)

# Directory of dpdk
# To use the library, must specify this path.
set(VENDOR_DPDK_DIR ${CMAKE_SOURCE_DIR}/vendor/dpdk)

add_subdirectory(module/gpuflow)

add_executable(GPUFlow module/main.cpp)

target_link_libraries(GPUFlow GPU_FLOW_LIBS)
