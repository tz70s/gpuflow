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
#include "cuda_sync_lcore_function.h"

namespace gpuflow {
namespace cu {

__device__ void IPv4Processing(struct ipv4_hdr *ipv4_header, struct ether_hdr *ether_header) {
  printf("Dealing with ipv4 header!\n");
}

__device__ void IPv6Processing(struct ipv6_hdr *ipv6_header, struct ether_hdr *ether_header) {
  printf("Dealing with ipv6 header!\n");
}

__device__ void EtherCopy(struct ether_hdr *ether_header) {

}

__global__ void PacketProcessing(uint8_t *dev_ptypes_burst,
                                 struct ipv4_hdr *dev_ipv4_hdr_burst,
                                 struct ipv6_hdr *dev_ipv6_hdr_burst,
                                 struct ether_hdr *dev_ether_hdrs_burst,
                                 int nb_of_ip_hdrs) {
  int idx = threadIdx.x;
  if (idx < nb_of_ip_hdrs) {
    // Match up packet types.
    if(dev_ptypes_burst[idx] == IP_FAMILY::PTYPE_IPV4) {
      IPv4Processing(&dev_ipv4_hdr_burst[idx], &dev_ether_hdrs_burst[idx]);
    } else if (dev_ptypes_burst[idx] == IP_FAMILY::PTYPE_IPV6) {
      IPv6Processing(&dev_ipv6_hdr_burst[idx], &dev_ether_hdrs_burst[idx]);
    } else {
      // ignore
    }
  }
}

inline void CUDA_MALLOC_WITH_FAIL(void **predicate, size_t size, const char *predicate_type) {
  if (cudaMalloc(predicate, size) != cudaSuccess) {
    std::cerr << "Device memory allocation on " << predicate_type << " failed, abort." << std::endl;
    exit(1);
  }
}

inline void CUDA_MEMCPY_WITH_FAIL(void *dst, const void *src, size_t size, cudaMemcpyKind kind,
                                  const char *operation_type) {
  if (cudaMemcpy(dst, src, size, kind) != cudaSuccess) {
    std::cerr << "Memory copy error on cuda mem copy of " << operation_type << std::endl;
    exit(1);
  }
}

inline int CudaSyncLCoreFunction::SetupCudaDevices(int nb_rx) {

  CUDA_MALLOC_WITH_FAIL((void **)&dev_ptypes_burst, nb_rx * sizeof(uint8_t), "dev_ptypes_burst");
  CUDA_MALLOC_WITH_FAIL((void **)&dev_ipv4_hdrs_burst, nb_rx * sizeof(struct ipv4_hdr), "dev_ipv4_hdrs_burst");
  CUDA_MALLOC_WITH_FAIL((void **)&dev_ipv6_hdrs_burst, nb_rx * sizeof(struct ipv6_hdr), "dev_ipv6_hdrs_burst");
  CUDA_MALLOC_WITH_FAIL((void **)&dev_ether_hdrs_burst, nb_rx * sizeof(struct ether_hdr), "dev_ether_hdrs_burst");

  return 0;
}

int CudaSyncLCoreFunction::ProcessPacketsBatch(struct rte_mbuf **pkts_burst, int nb_rx) {
  SetupCudaDevices(nb_rx);
  for (int i = 0; i < nb_rx; ++i) {
    if (RTE_ETH_IS_IPV4_HDR(pkts_burst[i]->packet_type)) {
      // Ipv4 header, copy ipv4
      CUDA_MEMCPY_WITH_FAIL(&dev_ipv4_hdrs_burst[i],
                            rte_pktmbuf_mtod_offset(pkts_burst[i], struct ipv4_hdr *, sizeof(struct ether_hdr)),
                            sizeof(struct ipv4_hdr),
                            cudaMemcpyHostToDevice,
                            "ipv4_header_memory_copy");

      // Add type into type burst.
      CUDA_MEMCPY_WITH_FAIL(&dev_ptypes_burst[i],
                            &IP_FAMILY::PTYPE_IPV4,
                            sizeof(uint8_t),
                            cudaMemcpyHostToDevice,
                            "ipv4_ptype_memory_copy");

      // Copy ether header
      CUDA_MEMCPY_WITH_FAIL(&dev_ether_hdrs_burst[i],
                            rte_pktmbuf_mtod(pkts_burst[i], struct ether_hdr *),
                            sizeof(struct ether_hdr),
                            cudaMemcpyHostToDevice,
                            "ipv4_ether_header_memory_copy");

    } else if (RTE_ETH_IS_IPV6_HDR(pkts_burst[i]->packet_type)) {
      // Ipv6 header, copy ipv6 type
      CUDA_MEMCPY_WITH_FAIL(&dev_ipv6_hdrs_burst[i],
                            rte_pktmbuf_mtod_offset(pkts_burst[i], struct ipv6_hdr *, sizeof(struct ether_hdr)),
                            sizeof(struct ipv6_hdr),
                            cudaMemcpyHostToDevice,
                            "ipv6_header_memory_copy");

      // Add type into type burst.
      CUDA_MEMCPY_WITH_FAIL(&dev_ptypes_burst[i],
                            &IP_FAMILY::PTYPE_IPV6,
                            sizeof(uint8_t),
                            cudaMemcpyHostToDevice,
                            "ipv6_ptype_memory_copy");

      // Copy ether header
      CUDA_MEMCPY_WITH_FAIL(&dev_ether_hdrs_burst[i],
                            rte_pktmbuf_mtod(pkts_burst[i], struct ether_hdr *),
                            sizeof(struct ether_hdr),
                            cudaMemcpyHostToDevice,
                            "ipv6_ether_header_memory_copy");
    } else {
      struct ether_hdr *ether_header = rte_pktmbuf_mtod(pkts_burst[i], struct ether_hdr *);
      if (ether_header->ether_type == rte_cpu_to_be_16(ETHER_TYPE_ARP)) {
        // is arp packet.
        std::cout << "ARP!" << std::endl;
      }
      // Continue, the index will be jumped off on either ethernet header burst, ipv4 header burst or ipv6 header burst.
    }
  }

  PacketProcessing<<<1, nb_rx>>>(dev_ptypes_burst,
          dev_ipv4_hdrs_burst,
          dev_ipv6_hdrs_burst,
          dev_ether_hdrs_burst,
          nb_rx);

  cudaFree(dev_ptypes_burst);
  cudaFree(dev_ether_hdrs_burst);
  cudaFree(dev_ipv4_hdrs_burst);
  cudaFree(dev_ipv6_hdrs_burst);
  return 0;
}

} // namespace cu
} // namespace gpuflow

