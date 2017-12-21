
file(GLOB_RECURSE CORE_SOURCES "*.cpp")
file(GLOB_RECURSE CORE_HEADERS "*.h")

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
        rte_pmd_i40e
        rte_pmd_vmxnet3_uio
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

add_library(GPU_FLOW_LIBS ${CORE_SOURCES} ${CORE_HEADERS} dataplane_processor.cpp dataplane_processor.h)

target_link_libraries(GPU_FLOW_LIBS ${CMAKE_THREAD_LIBS_INIT} ${DPDK_LIBS} ${CMAKE_DL_LIBS} dl rt)
