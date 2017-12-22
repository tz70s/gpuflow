/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include "dataplane.h"

#include <iostream>
#include <rte_mempool.h>

namespace gpuflow {

DataPlane::DataPlane(int argc, char *argv[], unsigned int num_of_cores) :
        num_of_cores(num_of_cores),
        pkt_mbuf_pool(nullptr) {
  // Initialize eal
  int ret = rte_eal_init(argc, argv);
  if (ret < 0) {
    rte_exit(EXIT_FAILURE, "ERROR with EAL initialization\n");
  }
  // Find binding eth devs
  int num_of_eth_devs = rte_eth_dev_count();
  if (num_of_eth_devs <= 0) {
    std::cerr << "Didn't find any eth devices, abort" << std::endl;
    exit(1);
  }
  // Create and initialize memory buffer pool
  if (CreateMbufPool() < 0) {
    std::cerr << "Error occurred on creating memory buffer pool of dpdk, abort" << std::endl;
    exit(1);
  }
}

int DataPlane::CreateMbufPool() {
  // Create mbuf pool
  pkt_mbuf_pool = rte_pktmbuf_pool_create("dataplane_mem_pool", NUM_BYTES_MBUF,
                                          MEMPOOL_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
  if (pkt_mbuf_pool == nullptr) {
    std::cerr << "Can't initialize memory buffer pool, rte_errno : " << rte_strerror(-rte_errno) << std::endl;
    return -1;
  }
  return 0;
}

void DataPlane::ServeProcessingLoop(int DataPlaneCore_t) {
  DataPlaneCore *data_plane_core;
  // Match Core type
  switch (DataPlaneCore_t) {
    case SayHelloCore_t:
      data_plane_core = new SayHelloCore();
      break;
    case DumpPacketCore_t:
      data_plane_core = new DumpPacketCore(pkt_mbuf_pool);
      break;
    default:
      std::cerr << "No matching Core, abort" << std::endl;
      exit(1);
  }
  data_plane_core->LCoreFunctions();
}

} // namespace gpuflow
