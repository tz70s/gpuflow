file(GLOB_RECURSE CORE_SOURCES "*.cpp")
file(GLOB_RECURSE CORE_HEADERS "*.h")

file(GLOB_RECURSE CORE_CPU_SOURCES "cpu/*.cpp")
file(GLOB_RECURSE CORE_CPU_HEADERS "cpu/*.h")

file(GLOB_RECURSE CORE_GPU_SOURCES "gpu/*.cpp")
file(GLOB_RECURSE CORE_GPU_HEADERS "gpu/*.h")

if (${CUDA_FOUND})
    # CU files
    file(GLOB_RECURSE CORE_GPU_CUDA_HEADERS "gpu/cuda/*.h")
    file(GLOB_RECURSE CORE_GPU_CUDA_SOURCES "gpu/cuda/*.cu")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_61,code=sm_61; -std=c++11)
    cuda_add_library(CU_OBJ ${CORE_GPU_CUDA_SOURCES})
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Set up dpdk libs
SET(DPDK_LIBS
        rte_net
        rte_ethdev
        rte_distributor
        rte_kni
        rte_mbuf
        rte_pmd_bond
        rte_pmd_ring
        rte_power
        rte_acl
        rte_eal
        rte_kvargs
        rte_mempool
        rte_mempool_ring
        rte_pmd_e1000
        rte_pmd_virtio
        rte_ring
        rte_cfgfile
        rte_hash
        rte_lpm
        rte_meter
        rte_pmd_vmxnet3_uio
        rte_pmd_tap
        rte_sched
        rte_timer
        rte_cmdline
        rte_ip_frag
        rte_pipeline
        rte_pmd_ixgbe
        rte_mempool_stack
        rte_port
        rte_table
        rte_pmd_bnxt
        rte_pmd_ena
        rte_pmd_nfp)

SET(ALL_LIBS
        -Wl,--whole-archive
        -Wl,--start-group ${DPDK_LIBS} numa -Wl,--end-group
        pthread dl rt
        -Wl,--no-whole-archive)

if (${CUDA_FOUND})
    # Compile cuda binary
    cuda_add_library(GPU_FLOW_LIBS
            ${CORE_SOURCES} ${CORE_HEADERS}
            ${CORE_CPU_SOURCES} ${CORE_CPU_HEADERS}
            ${CORE_GPU_SOURCES} ${CORE_GPU_HEADERS})
else()
    add_library(GPU_FLOW_LIBS
            ${CORE_SOURCES} ${CORE_HEADERS}
            ${CORE_CPU_SOURCES} ${CORE_CPU_HEADERS})
endif ()

target_link_libraries(GPU_FLOW_LIBS ${ALL_LIBS})

