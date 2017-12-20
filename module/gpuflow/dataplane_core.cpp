/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include "dataplane_core.h"

#include <iostream>

namespace gpuflow {

DataPlaneCore::DataPlaneCore(int argc, char **argv, unsigned int num_of_cores) : num_of_cores(num_of_cores),
                                                                                 pkt_mbuf_pool(nullptr) {
  int ret = rte_eal_init(argc, argv);
  if (ret < 0) {
    rte_exit(EXIT_FAILURE, "ERROR with EAL initialization\n");
  }

  // Create and initialize memory buffer pool
  if (CreateMbufPool() < 0) {
    exit(1);
  }
}

int DataPlaneCore::CreateMbufPool() {
  // Create mbuf pool
  pkt_mbuf_pool = rte_pktmbuf_pool_create("packet_mbuf_pool", NUM_BYTES_MBUF,
                                          MEMPOOL_CACHE_SIZE, 0, MBUF_DATA_SIZE, rte_socket_id());
  if (pkt_mbuf_pool == nullptr) {
    std::cerr << "Can't initialize memory buffer pool" << std::endl;
    return -1;
  }
}

} // namespace gpuflow
