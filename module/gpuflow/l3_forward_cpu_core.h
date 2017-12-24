/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _LPMV4_CPU_CORE_H_
#define _LPMV4_CPU_CORE_H_

#include <rte_lpm.h>
#include <rte_ip.h>
#include "dataplane_core.h"

namespace gpuflow {

namespace route {

struct IPv4LPMRoute {
  uint32_t ip;
  uint8_t depth;
  uint8_t if_out;
};

} // namespace route

class L3ForwardCPUCore : public DataPlaneCore {
 public:
  L3ForwardCPUCore(std::vector<ether_addr> *mac_addresses_ptr);
  void LCoreFunctions() override;

 private:
  std::vector<ether_addr> *mac_addresses_ptr;
  unsigned const int MAX_LPM_IPV4_RULES = 1024;
  route::IPv4LPMRoute ipv4_lpm_route_array[5];
  void CreateIPv4LPMRouteArray();
  void CreateLPMTable(int socket_id);
  inline uint16_t LPMLookUp(ipv4_hdr *ipv4_header, uint16_t port_id, int socket_id);
  void SimpleLPMForward(rte_mbuf *mbuf, unsigned int port_id, int socket_id);
  rte_lpm *ipv4_lpm_lookup_struct[1];
};

} // namespace gpuflow

#endif // _LPMV4_CPU_CORE_H_
