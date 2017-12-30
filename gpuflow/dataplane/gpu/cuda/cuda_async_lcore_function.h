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

struct ProcessingBatchFrame {
 public:
  explicit ProcessingBatchFrame(uint8_t batch_size);
  CustomEtherIPHeader *dev_custom_ether_ip_headers_burst;
  struct rte_mbuf **pkts_burst;
  cudaStream_t cuda_stream[32];
  uint8_t batch_size;
  uint8_t *dev_dst_ports_burst;
  uint8_t host_dst_ports_burst[32] = { (uint8_t) 254};
  uint8_t busy_counter;
};

struct SendFrame {
  SendFrame(uint8_t index, ProcessingBatchFrame *batch_frame, uint8_t port_id)
          : index(index), batch_frame_ptr(batch_frame), self_port(port_id) {}
  uint8_t index;
  ProcessingBatchFrame *batch_frame_ptr;
  uint8_t self_port;
};

class CudaASyncLCoreFunction {
 public:
  CudaASyncLCoreFunction(uint8_t _port_id, unsigned int _num_of_eth_devs,
                         std::vector<ether_addr> *_mac_addresses_ptr, IPv4RuleEntry *_lpm_table_ptr);
  int SetupCudaDevices();
  void CreateProcessingBatchFrame(int num_of_batch, uint8_t batch_size);
  int ProcessPacketsBatch(int batch_idx, struct rte_mbuf **pkts_burst, int nb_rx);
 private:
  ProcessingBatchFrame **batch_head;
  uint8_t port_id;
  unsigned int num_of_eth_devs;
  std::vector<ether_addr> *mac_addresses_ptr;
  ether_addr *dev_mac_addresses_array;
  IPv4RuleEntry *lpm_table_ptr;
};

} // namespace cu
} // namespace gpuflow

#endif // _ASYNC_LCORE_FUNCTION_CU_H_
