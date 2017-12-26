/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _L3_FORWARD_GPU_CORE_H_
#define _L3_FORWARD_GPU_CORE_H_

#include <rte_lpm.h>
#include <rte_ip.h>
#include <rte_ether.h>
#include <dataplane/dataplane_lpm_gpu.h>
#include "dataplane/dataplane_core.h"
#include "dataplane/dataplane_lpm.h"

namespace gpuflow {

class L3ForwardGPUCore : public DataPlaneCore {
 public:
  explicit L3ForwardGPUCore(unsigned int num_of_eth_devs, std::vector<ether_addr> *mac_addresses_ptr);
  void LCoreFunctions() override;

 private:
  unsigned int num_of_eth_devs;
  std::vector<ether_addr> *mac_addresses_ptr;
  DataPlaneLPMIPv4GPU data_plane_lpm_ipv4_gpu;
};

} // namespace gpuflow

#endif // _L3_FORWARD_GPU_CORE_H_
