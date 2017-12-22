/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _DATAPLANE_Core_H_
#define _DATAPLANE_Core_H_

#include <rte_launch.h>
#include <vector>
#include <string>
#include <rte_mempool.h>

namespace gpuflow {

enum {
  SayHelloCore_t,
  DumpPacketCore_t
};

class DataPlaneCore {
 public:
  virtual void LCoreFunctions() = 0;
};

class SayHelloCore : public DataPlaneCore {
 public:
  void LCoreFunctions() override;
};

class DumpPacketCore : public DataPlaneCore {
 public:
  explicit DumpPacketCore(rte_mempool *pkt_mbuf_pool)
          : pkt_mbuf_pool(pkt_mbuf_pool) {};

  void LCoreFunctions() override;

 private:
  // FIXME: Thread safe?
  rte_mempool *pkt_mbuf_pool;
};
} // namespace gpuflow

#endif // _DATAPLANE_Core_H_
