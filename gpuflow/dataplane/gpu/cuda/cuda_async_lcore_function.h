/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _ASYNC_LCORE_FUNCTION_CU_H_
#define _ASYNC_LCORE_FUNCTION_CU_H_

#include <vector>
#include <rte_ethdev.h>
#include "cuda_lpm_factory.h"

namespace gpuflow {
namespace cu {

namespace IP_FAMILY {

const uint8_t PTYPE_IPV4 = 0x12;
const uint8_t PTYPE_IPV6 = 0x13;

} // namespace IP_FAMILY

extern "C" {

// Use this for force alignment.
struct CustomEtherIPHeader {
  struct ether_hdr ether_header;
  struct ipv6_hdr ipv6_header;
} __attribute__((__packed__));

}

class CudaASyncLCoreFunction {
 public:
  CudaASyncLCoreFunction(unsigned int num_of_eth_devs, std::vector<ether_addr> *mac_addresses_ptr)
          : num_of_eth_devs(num_of_eth_devs), mac_addresses_ptr(mac_addresses_ptr), head(0), tail(255) {}

  int SetupCudaDevices();
  int ProcessPacketsBatch(struct rte_mbuf **pkts_burst, int nb_rx, IPv4RuleEntry *lpm_table_ptr);
  ~CudaASyncLCoreFunction() {
    cudaFree(dev_custom_ether_ip_headers_ring);
  }
 private:
  unsigned int num_of_eth_devs;
  std::vector<ether_addr> *mac_addresses_ptr;
  uint8_t head;
  uint8_t tail;
  CustomEtherIPHeader *dev_custom_ether_ip_headers_ring;

};

} // namespace cu
} // namespace gpuflow

#endif // _ASYNC_LCORE_FUNCTION_CU_H_
