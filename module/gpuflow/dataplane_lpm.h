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

// Abstract class for defining common member and methods for a LPM object.
// It's not used for dynamic (subtype) polymorphism, but common share member can be parameterized.
template <typename LPMRoute, typename LookUpType, typename IP_Header>
class AbstractDataPlaneLPM {
 public:
  virtual uint16_t RoutingTableLookUp(IP_Header *ip_header, uint16_t port_id, int socket_id) = 0;

 protected:
  unsigned const int MAX_LPM_ROUTING_RULES = 1024;
  LPMRoute ip_lpm_route_array[5];
  LookUpType *ip_lpm_lookup_struct[1];
  virtual void CreateLPMTable(int socket_id) = 0;
};

class DataPlaneLPMv4 : public AbstractDataPlaneLPM<route::IPv4LPMRoute, struct rte_lpm, ipv4_hdr> {
 public:
  DataPlaneLPMv4();
  uint16_t RoutingTableLookUp(ipv4_hdr *ipv4_header, uint16_t port_id, int socket_id) override ;

 private:
  void CreateLPMTable(int socket_id) override;
};

class DataPlaneLPMv6 : public AbstractDataPlaneLPM<route::IPv6LPMRoute, struct rte_lpm6, ipv6_hdr> {
 public:
  DataPlaneLPMv6();
  uint16_t RoutingTableLookUp(ipv6_hdr *ipv6_header, uint16_t port_id, int socket_id) override ;

 private:
  void CreateLPMTable(int socket_id) override;
};

} // namespace gpuflow

#endif // _DATAPLANE_LPM_H_
