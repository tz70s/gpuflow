/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _DATAPLANE_PROCESSOR_H_
#define _DATAPLANE_PROCESSOR_H_

#include <rte_launch.h>
#include <vector>
#include <string>

namespace gpuflow {

enum {
  SayHelloProcessor_t,
  DumpPacketProcessor_t
};

class DataPlaneProcessor {
 public:
  virtual void LCoreFunctions() = 0;
};

class SayHelloProcessor : public DataPlaneProcessor {
 public:
  void LCoreFunctions() override;
};

class DumpPacketProcessor : public DataPlaneProcessor {
 public:
  explicit DumpPacketProcessor(std::vector<int> *tap_fds, struct rte_mempool *pkt_mbuf_pool)
          : tap_fds(tap_fds), pkt_mbuf_pool(pkt_mbuf_pool) {};

  void LCoreFunctions() override;

 private:
  std::vector<int> *tap_fds;

  // FIXME: Thread safe?
  struct rte_mempool *pkt_mbuf_pool;
};
} // namespace gpuflow

#endif // _DATAPLANE_PROCESSOR_H_
