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

__device__ void EtherCopy(struct ether_hdr *ether_header, uint8_t port_id, uint8_t dst_port,
                          ether_addr *dev_mac_addresses_array) {
  // Change source address of this port's
  memcpy(&ether_header->s_addr, &dev_mac_addresses_array[port_id], sizeof(ether_addr));
  // Change dst address of dst port's
  memcpy(&ether_header->d_addr, &dev_mac_addresses_array[dst_port], sizeof(ether_addr));
}

__device__ void IPv4Processing(CustomEtherIPHeader *custom_ether_ip_header, IPv4RuleEntry *lpm_table_ptr,
                               uint8_t port_id, ether_addr *dev_mac_addresses_array) {

  // Force cast to ipv4 header
  struct ipv4_hdr *ipv4_header = (struct ipv4_hdr *)&custom_ether_ip_header->ipv6_header;

  // Transferring endian, 32-bit
  uint32_t ipv4_addr_le = ((ipv4_header->dst_addr >> 24) & 0xff) |
          ((ipv4_header->dst_addr << 8) & 0xff0000) |
          ((ipv4_header->dst_addr >> 8) & 0xff00) |
          ((ipv4_header->dst_addr << 24) & 0xff000000);

  if ((lpm_table_ptr + (ipv4_addr_le >> 16)) != nullptr) {
    IPv4RuleEntry *entry = (lpm_table_ptr + (ipv4_addr_le >> 16));
    if (entry->valid_flag) {
      EtherCopy(&custom_ether_ip_header->ether_header, port_id, entry->next_hop, dev_mac_addresses_array);
      custom_ether_ip_header->dst_port = entry->next_hop;
    } else {
      custom_ether_ip_header->dst_port = 254;
    }
  }
}

__device__ void IPv6Processing(CustomEtherIPHeader *custom_ether_ip_header) {
  // TODO : Add ipv6 processing
}

__global__ void PacketProcessing(CustomEtherIPHeader *dev_custom_ether_ip_header_burst,
                                 uint8_t port_id,
                                 IPv4RuleEntry *lpm_table_ptr,
                                 ether_addr *dev_mac_addresses_array,
                                 int nb_of_ip_hdrs) {
  int idx = threadIdx.x;
  if (idx < nb_of_ip_hdrs) {
    // Match up packet types.
    if(dev_custom_ether_ip_header_burst[idx].ether_header.ether_type ==
            (((ETHER_TYPE_IPv4 >> 8) | (ETHER_TYPE_IPv4 << 8)) & 0xffff)) {
      // IPv4 header
      IPv4Processing(&dev_custom_ether_ip_header_burst[idx], lpm_table_ptr, port_id, dev_mac_addresses_array);
    } else if (dev_custom_ether_ip_header_burst[idx].ether_header.ether_type ==
            (((ETHER_TYPE_IPv6 >> 8) | (ETHER_TYPE_IPv6 << 8)) & 0xffff)) {
      // IPv6 header
      IPv6Processing(&dev_custom_ether_ip_header_burst[idx]);
      dev_custom_ether_ip_header_burst[idx].dst_port = 254;
    } else if (dev_custom_ether_ip_header_burst[idx].ether_header.ether_type ==
            (((ETHER_TYPE_ARP >> 8) | (ETHER_TYPE_ARP << 8)) & 0xffff)){
      // Send to all
      dev_custom_ether_ip_header_burst[idx].dst_port = 255;
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
  CudaMallocWithFailOver((void **) &dev_mac_addresses_array, num_of_eth_devs * sizeof(struct ether_addr),
                         "dev_mac_addresses_array");
  // Copy mac addresses into device memory
  cudaStream_t mac_stream;
  cudaStreamCreate(&mac_stream);
  unsigned int count = 0;
  for (auto it = mac_addresses_ptr->begin(); it != mac_addresses_ptr->end(); ++it) {
    CudaASyncMemcpyWithFailOver(&dev_mac_addresses_array[count++],
                                &(*it),
                                sizeof(ether_addr),
                                cudaMemcpyHostToDevice,
                                mac_stream, "dev_mac_addresses_array_memory_copy");
  }
  cudaDeviceSynchronize();
  return 0;
}

ProcessingBatchFrame::ProcessingBatchFrame(uint8_t _batch_size) : batch_size(_batch_size),
                                                                  busy(false) {
  cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
  cudaMallocHost((void **) &host_custom_ether_ip_headers_burst, batch_size * sizeof(CustomEtherIPHeader));
  for (uint8_t i = 0; i < batch_size; i++) {
    host_custom_ether_ip_headers_burst[i].dst_port = 254;
  }
  CudaMallocWithFailOver((void **) &dev_custom_ether_ip_headers_burst, batch_size * sizeof(CustomEtherIPHeader),
                         "dev_custom_ether_ip_headers_burst");
}

void CudaASyncLCoreFunction::CreateProcessingBatchFrame(int num_of_batch, uint8_t batch_size){
  batch_head = new ProcessingBatchFrame *[num_of_batch];
  for (int i = 0; i < num_of_batch; i++) {
    batch_head[i] = new ProcessingBatchFrame(batch_size);
  }
}

int CudaASyncLCoreFunction::ProcessPacketsBatch(ProcessingBatchFrame *self_batch) {

  CudaASyncMemcpyWithFailOver(self_batch->dev_custom_ether_ip_headers_burst,
                              self_batch->host_custom_ether_ip_headers_burst,
                              self_batch->nb_rx * sizeof(CustomEtherIPHeader),
                              cudaMemcpyHostToDevice,
                              self_batch->cuda_stream,
                              "custom_ether_ip_header_memory_copy");

  PacketProcessing<<<1, self_batch->nb_rx, 0, self_batch->cuda_stream>>>(self_batch->dev_custom_ether_ip_headers_burst,
          port_id,
          lpm_table_ptr,
          dev_mac_addresses_array,
          self_batch->nb_rx);

  CudaASyncMemcpyWithFailOver(self_batch->host_custom_ether_ip_headers_burst,
                              self_batch->dev_custom_ether_ip_headers_burst,
                              self_batch->nb_rx * sizeof(CustomEtherIPHeader),
                              cudaMemcpyDeviceToHost,
                              self_batch->cuda_stream,
                              "custom_ether_header_memory_copy_back");

  cudaStreamAddCallback(self_batch->cuda_stream, [](cudaStream_t stream, cudaError_t status, void *self_batch_ptr) {
    auto *self_batch = (ProcessingBatchFrame *) self_batch_ptr;
    self_batch->ready_to_burst = true;
  }, self_batch, 0);

  return 0;
}

CudaASyncLCoreFunction::CudaASyncLCoreFunction(uint8_t _port_id, unsigned int _num_of_eth_devs,
                                               std::vector<ether_addr> *_mac_addresses_ptr, IPv4RuleEntry *_lpm_table_ptr)
        : port_id(_port_id), num_of_eth_devs(_num_of_eth_devs), mac_addresses_ptr(_mac_addresses_ptr),
          lpm_table_ptr(_lpm_table_ptr) {
  // Do nothing
}

} // namespace cu
} // namespace gpuflow

