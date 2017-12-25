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
  DumpPacketCore_t,
  BasicForwardCore_t,
  L3ForwardCPUCore_t,
};

class DataPlaneCore {
 public:
  virtual void LCoreFunctions() = 0;

  virtual ~DataPlaneCore() = default;
};

class SayHelloCore : public DataPlaneCore {
 public:
  void LCoreFunctions() override;
};

class DumpPacketCore : public DataPlaneCore {
 public:
  DumpPacketCore() = default;

  void LCoreFunctions() override;
};

class BasicForwardCore : public DataPlaneCore {
 public:
  BasicForwardCore() = default;
  void LCoreFunctions() override;
};

} // namespace gpuflow

#endif // _DATAPLANE_Core_H_
