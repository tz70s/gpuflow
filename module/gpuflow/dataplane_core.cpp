/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include "dataplane_core.h"

#include <iostream>
#include <rte_mempool.h>

namespace gpuflow {

DataPlaneCore::DataPlaneCore(int argc, char *argv[], DataPlane *data_plane_ptr, unsigned int num_of_cores) :
        num_of_cores(num_of_cores),
        tap_fds(data_plane_ptr->retrieve_tap_fds()), pkt_mbuf_pool(nullptr) {
  int ret = rte_eal_init(argc, argv);
  if (ret < 0) {
    rte_exit(EXIT_FAILURE, "ERROR with EAL initialization\n");
  }
  // Create and initialize memory buffer pool
  if (CreateMbufPool() < 0) {
    std::cerr << "Error occurred on creating memory buffer pool of dpdk, abort" << std::endl;
    exit(1);
  }
}

int DataPlaneCore::CreateMbufPool() {
  // Create mbuf pool
  pkt_mbuf_pool = rte_pktmbuf_pool_create("dataplane_mem_pool", NUM_BYTES_MBUF,
                                          MEMPOOL_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
  if (pkt_mbuf_pool == nullptr) {
    std::cerr << "Can't initialize memory buffer pool, rte_errno : " << rte_strerror(-rte_errno) << std::endl;
    return -1;
  }
  return 0;
}

void DataPlaneCore::ServeProcessingLoop(int DataPlaneProcessor_t) {
  DataPlaneProcessor *data_plane_processor;
  // Match processor type
  switch (DataPlaneProcessor_t) {
    case SayHelloProcessor_t:
      data_plane_processor = new SayHelloProcessor();
      break;
    case DumpPacketProcessor_t:
      data_plane_processor = new DumpPacketProcessor(tap_fds, pkt_mbuf_pool);
      break;
    default:
      std::cerr << "No matching processor, abort" << std::endl;
      exit(1);
  }
  data_plane_processor->LCoreFunctions();
}

} // namespace gpuflow
