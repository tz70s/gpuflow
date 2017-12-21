/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _DATAPLANE_CORE_H_
#define _DATAPLANE_CORE_H_

#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_config.h>
#include "dataplane_processor.h"

namespace gpuflow {

class DataPlaneCore {
 public:
  DataPlaneCore(int argc, char *argv[], unsigned int num_of_cores = 4);

  void ServeProcessingLoop(lcore_function_t *);

 private:
  unsigned int num_of_cores;
  unsigned const int NUM_BYTES_MBUF = 8192;
  unsigned const int MEMPOOL_CACHE_SIZE = 256;

  int CreateMbufPool();
  rte_mempool *pkt_mbuf_pool;

};

} // namespace gpuflow

#endif // _DATAPLANE_CORE_H_
