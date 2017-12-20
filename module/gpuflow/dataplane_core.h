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

namespace gpuflow {

class DataPlaneCore {
 public:
  DataPlaneCore(int argc, char *argv[], unsigned int num_of_cores = 4);

 private:
  unsigned int num_of_cores;
  unsigned const int NUM_BYTES_MBUF = 8192;
  unsigned const int MEMPOOL_CACHE_SIZE = 32;
  unsigned const short MAX_PACKET_SIZE = 2048;
  unsigned const short MBUF_DATA_SIZE = MAX_PACKET_SIZE + RTE_PKTMBUF_HEADROOM;

  int CreateMbufPool();
  rte_mempool *pkt_mbuf_pool;

};

} // namespace gpuflow

#endif // _DATAPLANE_CORE_H_
