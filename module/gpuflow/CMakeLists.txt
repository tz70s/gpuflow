
file(GLOB_RECURSE CORE_SOURCES "*.cpp")
file(GLOB_RECURSE CORE_HEADERS "*.h")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Set up dpdk libs
SET(DPDK_LIBS
        dpdk              rte_mempool_stack         rte_pmd_nfp
        rte_acl           rte_meter                 rte_pmd_null
        rte_bitratestats  rte_metrics               rte_pmd_null_crypto
        rte_cfgfile       rte_net                   rte_pmd_octeontx_ssovf
        rte_cmdline       rte_pdump                 rte_pmd_qede
        rte_cryptodev     rte_pipeline              rte_pmd_ring
        rte_distributor   rte_pmd_af_packet         rte_pmd_sfc_efx
        rte_eal           rte_pmd_ark               rte_pmd_skeleton_event
        rte_efd           rte_pmd_avp               rte_pmd_sw_event
        rte_ethdev        rte_pmd_bnxt              rte_pmd_tap
        rte_eventdev      rte_pmd_bond              rte_pmd_thunderx_nicvf
        rte_gro           rte_pmd_crypto_scheduler  rte_pmd_vhost
        rte_hash          rte_pmd_cxgbe             rte_pmd_virtio
        rte_ip_frag       rte_pmd_e1000             rte_pmd_vmxnet3_uio
        rte_jobstats      rte_pmd_ena               rte_port
        rte_kni           rte_pmd_enic              rte_power
        rte_kvargs        rte_pmd_failsafe          rte_reorder
        rte_latencystats  rte_pmd_fm10k             rte_ring
        rte_lpm           rte_pmd_i40e              rte_sched
        rte_mbuf          rte_pmd_ixgbe             rte_table
        rte_mempool       rte_pmd_kni               rte_timer
        rte_mempool_ring  rte_pmd_lio               rte_vhost)


add_library(GPU_FLOW_LIBS ${CORE_SOURCES} ${CORE_HEADERS} dataplane_processor.cpp dataplane_processor.h)

target_link_libraries(GPU_FLOW_LIBS ${CMAKE_THREAD_LIBS_INIT} ${DPDK_LIBS} ${CMAKE_DL_LIBS} dl rt ${NUMA_LIBRARY})
