/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _DATAPLANE_H_
#define _DATAPLANE_H_

#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_config.h>
#include <rte_eal.h>
#include "dataplane_core.h"
#include <vector>

namespace gpuflow {

class DataPlane {
 public:
  DataPlane(int argc, char *argv[], unsigned int num_of_cores = 4);
  void ServeProcessingLoop(int);
  std::vector<ether_addr> mac_addresses;

 private:
  unsigned int num_of_cores;
  unsigned int NUM_BYTES_MBUF = 1024;
  unsigned int MEMPOOL_CACHE_SIZE = 32;

  struct rte_eth_conf port_conf = {};
  int CreateMbufPool();
  void InitializePortConf();
  int InitializePorts(uint8_t port);

  rte_mempool *pkt_mbuf_pool;

};

} // namespace gpuflow

#endif // _DATAPLANE_H_
