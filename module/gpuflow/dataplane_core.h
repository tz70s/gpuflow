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
#include <rte_eal.h>
#include "dataplane_processor.h"
#include "dataplane.h"

namespace gpuflow {

// Forward declaration, for circular dependency.
class DataPlane;

class DataPlaneCore {
 public:
  DataPlaneCore(int argc, char *argv[], DataPlane *data_plane_ptr, unsigned int num_of_cores = 4);
  void ServeProcessingLoop(int);

 private:
  unsigned int num_of_cores;
  unsigned int NUM_BYTES_MBUF = 1024;
  unsigned int MEMPOOL_CACHE_SIZE = 32;

  std::vector<int> *tap_fds;

  int CreateMbufPool();
  struct rte_mempool *pkt_mbuf_pool;

};

} // namespace gpuflow

#endif // _DATAPLANE_CORE_H_
