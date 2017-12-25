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
#include "gpuflow/dataplane_core.h"
#include "gpuflow/dataplane_lpm.h"

namespace gpuflow {

class L3ForwardGPUCore : public DataPlaneCore {
 public:
  explicit L3ForwardGPUCore(std::vector<ether_addr> *mac_addresses_ptr);
  void LCoreFunctions() override;

 private:
  std::vector<ether_addr> *mac_addresses_ptr;
  DataPlaneLPMv4 data_plane_lpm_v4;
  DataPlaneLPMv6 data_plane_lpm_v6;
  void SimpleLPMForward(rte_mbuf *mbuf, unsigned int port_id, int socket_id);

};

} // namespace gpuflow

#endif // _L3_FORWARD_GPU_CORE_H_
