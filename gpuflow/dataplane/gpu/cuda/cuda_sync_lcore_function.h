/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _SYNC_LCORE_FUNCTION_CU_H_
#define _SYNC_LCORE_FUNCTION_CU_H_

#include <vector>
#include <rte_ethdev.h>
namespace gpuflow {
namespace cu {

namespace IP_FAMILY {

const uint8_t PTYPE_IPV4 = 0x12;
const uint8_t PTYPE_IPV6 = 0x13;
}

class CudaSyncLCoreFunction {
 public:
  CudaSyncLCoreFunction(unsigned int num_of_eth_devs, std::vector<ether_addr> *mac_addresses_ptr)
          : num_of_eth_devs(num_of_eth_devs), mac_addresses_ptr(mac_addresses_ptr) {}

  int SetupCudaDevices(int nb_rx);
  int ProcessPacketsBatch(struct rte_mbuf **pkts_burst, int nb_rx);
 private:
  unsigned int num_of_eth_devs;
  std::vector<ether_addr> *mac_addresses_ptr;
  uint8_t *dev_ptypes_burst;
  struct ether_hdr *dev_ether_hdrs_burst;
  struct ipv4_hdr *dev_ipv4_hdrs_burst;
  struct ipv6_hdr *dev_ipv6_hdrs_burst;

};

} // namespace cu
} // namespace gpuflow

#endif // _SYNC_LCORE_FUNCTION_CU_H_
