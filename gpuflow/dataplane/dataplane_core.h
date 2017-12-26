/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _DATAPLANE_CORE_H_
#define _DATAPLANE_CORE_H_

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
	L3ForwardGPUCore_t
};

class DataPlaneCore {
 public:
  explicit DataPlaneCore(unsigned int num_of_eth_devs) : num_of_eth_devs(num_of_eth_devs) {}
  unsigned int num_of_eth_devs;
  virtual void LCoreFunctions() = 0;

  virtual ~DataPlaneCore() = default;
};

class SayHelloCore : public DataPlaneCore {
 public:
  explicit SayHelloCore(unsigned int num_of_eth_devs) : DataPlaneCore(num_of_eth_devs) {}
  void LCoreFunctions() override;
};

class DumpPacketCore : public DataPlaneCore {
 public:
  explicit DumpPacketCore(unsigned int num_of_eth_devs) : DataPlaneCore(num_of_eth_devs) {}

  void LCoreFunctions() override;
};

class BasicForwardCore : public DataPlaneCore {
 public:
  explicit BasicForwardCore(unsigned int num_of_eth_devs) : DataPlaneCore(num_of_eth_devs) {}
  void LCoreFunctions() override;
};

} // namespace gpuflow

#endif // _DATAPLANE_CORE_H_
