/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _LPM_CPU_CORE_H_
#define _LPM_CPU_CORE_H_

#include <rte_lpm.h>
#include <rte_ip.h>
#include "gpuflow/dataplane_core.h"
#include "gpuflow/dataplane_lpm.h"

namespace gpuflow {

class L3ForwardCPUCore : public DataPlaneCore {
 public:
  explicit L3ForwardCPUCore(std::vector<ether_addr> *mac_addresses_ptr);
  void LCoreFunctions() override;

 private:
  std::vector<ether_addr> *mac_addresses_ptr;
  DataPlaneLPMv4 data_plane_lpm_v4;
  DataPlaneLPMv6 data_plane_lpm_v6;
  void SimpleLPMForward(rte_mbuf *mbuf, unsigned int port_id, int socket_id);

};

} // namespace gpuflow

#endif // _LPM_CPU_CORE_H_
