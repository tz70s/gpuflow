cmake_minimum_required(VERSION 3.5)
project(GPUFlow C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(numa REQUIRED)

SET(CMAKE_CXX_FLAGS "-fno-stack-protector -Wall -Wextra -Wno-unused-parameter -g -O3 -std=gnu++11 -march=native -msse4.2")
SET(CMAKE_C_FLAGS "-fno-stack-protector -Wall -Wextra -Wno-unused-parameter -g -O3 -std=gnu11 -march=native -msse4.2")

# For cross-platform compilation
if(EXISTS ${CMAKE_SOURCE_DIR}/vendor/dpdk-08)
  set(VENDOR_DPDK_DIR ${CMAKE_SOURCE_DIR}/vendor/dpdk-08)
else()
  # Directory of dpdk
  set(VENDOR_DPDK_DIR $ENV{RTE_SDK})
endif()
include_directories(${VENDOR_DPDK_DIR}/x86_64-native-linuxapp-gcc/include)
link_directories(${VENDOR_DPDK_DIR}/x86_64-native-linuxapp-gcc/lib)

add_subdirectory(module/gpuflow)

add_executable(GPUFlow module/main.cpp)

target_link_libraries(GPUFlow GPU_FLOW_LIBS)
