/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include <stdio.h>
#include <rte_mbuf.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <rte_ip.h>
#include "cuda_async_lcore_function.h"
#include "cuda_lpm_factory.h"

namespace gpuflow {
namespace cu {

__device__ void IPv4Processing(CustomEtherIPHeader *custom_ether_ip_header, IPv4RuleEntry *lpm_table_ptr) {

  // Force cast to ipv4 header
  struct ipv4_hdr *ipv4_header = (struct ipv4_hdr *)&custom_ether_ip_header->ipv6_header;

  // Transferring endian, 32-bit
  uint32_t ipv4_addr_le = ((ipv4_header->dst_addr >> 24) & 0xff) |
          ((ipv4_header->dst_addr << 8) & 0xff0000) |
          ((ipv4_header->dst_addr >> 8) & 0xff00) |
          ((ipv4_header->dst_addr << 24) & 0xff000000);

  printf("IPv4 dst address : %d\n", ipv4_addr_le >> 16);
  if ((lpm_table_ptr + (ipv4_addr_le >> 16)) != nullptr) {
    IPv4RuleEntry *entry = (lpm_table_ptr + (ipv4_addr_le >> 16));
    if (entry->valid_flag) {
      printf("Get the next hop! %d\n", entry->next_hop);
    }
  }
}

__device__ void IPv6Processing(CustomEtherIPHeader *custom_ether_ip_header) {
  printf("Dealing with ipv6 header!\n");
}

__device__ void EtherCopy(struct ether_hdr *ether_header) {

}

__global__ void PacketProcessing(CustomEtherIPHeader *dev_custom_ether_ip_header_ring,
                                 int nb_of_ip_hdrs,
                                 IPv4RuleEntry *lpm_table_ptr) {
  int idx = threadIdx.x;
  if (idx < nb_of_ip_hdrs) {
    // Match up packet types.
    if(dev_custom_ether_ip_header_ring[idx].ether_header.ether_type ==
            (((ETHER_TYPE_IPv4 >> 8) | (ETHER_TYPE_IPv4 << 8)) & 0xffff)) {
      // IPv4 header
      IPv4Processing(&dev_custom_ether_ip_header_ring[idx], lpm_table_ptr);
    } else if (dev_custom_ether_ip_header_ring[idx].ether_header.ether_type ==
            (((ETHER_TYPE_IPv6 >> 8) | (ETHER_TYPE_IPv6 << 8)) & 0xffff)) {
      // IPv6 header
      IPv6Processing(&dev_custom_ether_ip_header_ring[idx]);
    } else {
      // ignore
    }
  }
}

static inline void CudaMallocWithFailOver(void **predicate, size_t size, const char *predicate_type) {
  cudaError_t error = cudaMalloc(predicate, size);
  if (error != cudaSuccess) {
    std::cerr << "Device memory allocation on " << predicate_type << " failed, abort." << std::endl;
    std::cerr << cudaGetErrorName(error) << " " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }
}

static inline void CudaASyncMemcpyWithFailOver(void *dst, const void *src, size_t size, cudaMemcpyKind kind,
                                       cudaStream_t stream, const char *operation_type) {
  cudaError_t error = cudaMemcpyAsync(dst, src, size, kind, stream);
  if (error != cudaSuccess) {
    std::cerr << "Async Memory copy error on " << operation_type << std::endl;
    std::cerr << cudaGetErrorName(error) << " " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }
}

int CudaASyncLCoreFunction::SetupCudaDevices() {
  CudaMallocWithFailOver((void **) &dev_custom_ether_ip_headers_ring, 256 * sizeof(CustomEtherIPHeader),
                         "dev_custom_ether_ip_headers_ring");

  return 0;
}

int CudaASyncLCoreFunction::ProcessPacketsBatch(struct rte_mbuf **pkts_burst, int nb_rx,
                                                IPv4RuleEntry *lpm_table_ptr) {
  cudaStream_t cuda_stream;
  cudaStreamCreate(&cuda_stream);
  for (uint8_t i = 0; i < nb_rx; ++i) {

    if (RTE_ETH_IS_IPV4_HDR(pkts_burst[i]->packet_type)) {
      CudaASyncMemcpyWithFailOver(&dev_custom_ether_ip_headers_ring[head],
                                  rte_pktmbuf_mtod(pkts_burst[i], struct ether_hdr *),
                                  sizeof(CustomEtherIPHeader),
                                  cudaMemcpyHostToDevice,
                                  cuda_stream,
                                  "custom_ether_ipv4_header_memory_copy");

    } else if (RTE_ETH_IS_IPV6_HDR(pkts_burst[i]->packet_type)) {
      // Ipv6 header, copy ipv6 type
      CudaASyncMemcpyWithFailOver(&dev_custom_ether_ip_headers_ring[head],
                                  rte_pktmbuf_mtod(pkts_burst[i], struct ether_hdr *),
                                  sizeof(CustomEtherIPHeader),
                                  cudaMemcpyHostToDevice,
                                  cuda_stream,
                                  "custom_ether_ipv6_header_memory_copy");

    } else {
      struct ether_hdr *ether_header = rte_pktmbuf_mtod(pkts_burst[i], struct ether_hdr *);
      if (ether_header->ether_type == rte_cpu_to_be_16(ETHER_TYPE_ARP)) {
        // is arp packet.
        std::cout << "ARP!" << std::endl;
      }
      // Continue, the index will be jumped off on either ethernet header burst, ipv4 header burst or ipv6 header burst.
    }
    head++;
  }

  PacketProcessing<<<1, nb_rx, 0, cuda_stream>>>(&dev_custom_ether_ip_headers_ring[head - nb_rx], nb_rx, lpm_table_ptr);
  // Move on
  tail += nb_rx;
  // TODO: Copy back.
  // TODO: After stream copy back, add a callback here, and fire tx transferring.

  return 0;
}

} // namespace cu
} // namespace gpuflow

