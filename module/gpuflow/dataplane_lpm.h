/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _DATAPLANE_LPM_H_
#define _DATAPLANE_LPM_H_

#include <cstdint>

namespace gpuflow {
namespace route {

struct IPv4LPMRoute {
  uint32_t ip;
  uint8_t depth;
  uint8_t if_out;
};

struct IPv6LPMRoute {
  uint8_t ip[16];
  uint8_t  depth;
  uint8_t  if_out;
};

} // namespace route

class DataPlaneLPMBase {
 protected:
  unsigned const int MAX_LPM_ROUTING_RULES = 1024;
};

class DataPlaneLPMv4 : public DataPlaneLPMBase {
 public:
  DataPlaneLPMv4();
  uint16_t RoutingTableLookUp(ipv4_hdr *ipv4_header, uint16_t port_id, int socket_id);

 private:
  struct rte_lpm *ipv4_lpm_lookup_struct[1];
  route::IPv4LPMRoute ipv4_lpm_route_array[5];
  void CreateLPMTable(int socket_id);

};

class DataPlaneLPMv6 : public DataPlaneLPMBase {
 public:
  DataPlaneLPMv6();
  uint16_t RoutingTableLookUp(ipv6_hdr *ipv6_header, uint16_t port_id, int socket_id);

 private:
  struct rte_lpm6 *ipv6_lpm_lookup_struct[1];
  route::IPv6LPMRoute ipv6_lpm_route_array[5];
  void CreateLPMTable(int socket_id);
};

} // namespace gpuflow

#endif // _DATAPLANE_LPM_H_
