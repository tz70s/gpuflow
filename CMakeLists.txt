cmake_minimum_required(VERSION 3.5)
project(GPUFlow C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(numa REQUIRED)
find_package(CUDA)

if(${CUDA_FOUND})
    message("Found CUDA, compilation will include gpu version.")
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_HOST_COMPILER g++)
    add_definitions(-D_GPU_EXEC)
else()
    message("CUDA not found, compilation will include only cpu version.")
endif()

add_definitions(-D_DEBUG)

SET(CMAKE_CXX_FLAGS "-fno-stack-protector -Wall -Wextra -Wno-unused-parameter -g -O3 -std=c++11 -march=native -msse4.2")
SET(CMAKE_C_FLAGS "-fno-stack-protector -Wall -Wextra -Wno-unused-parameter -g -O3 -std=c11 -march=native -msse4.2")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# For cross-platform compilation
if(EXISTS ${CMAKE_SOURCE_DIR}/vendor/dpdk)
  set(VENDOR_DPDK_DIR ${CMAKE_SOURCE_DIR}/vendor/dpdk)
else()
  # Directory of dpdk
  set(VENDOR_DPDK_DIR $ENV{RTE_SDK})
endif()

include_directories(${VENDOR_DPDK_DIR}/x86_64-native-linuxapp-gcc/include)
link_directories(${VENDOR_DPDK_DIR}/x86_64-native-linuxapp-gcc/lib)

include_directories(gpuflow)
add_subdirectory(gpuflow/dataplane)

if(${CUDA_FOUND})
    cuda_add_executable(GPUFlow gpuflow/main.cpp)
    target_link_libraries(GPUFlow GPU_FLOW_LIBS CU_OBJ)
else()
    remove_definitions(-D_GPU_EXEC)
    add_executable(GPUFlow gpuflow/main.cpp)
    target_link_libraries(GPUFlow GPU_FLOW_LIBS)
endif()
