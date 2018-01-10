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

__device__ void IPv4Processing(CustomEtherIPHeader *custom_ether_ip_header, IPv4RuleEntry *lpm4_table_ptr,
                               uint8_t port_id, ether_addr *dev_mac_addresses_array, unsigned int num_of_eth_devs) {

  // Force cast to ipv4 header
  struct ipv4_hdr *ipv4_header = (struct ipv4_hdr *)&custom_ether_ip_header->ipv6_header;

  // Transferring endian, 32-bit
  uint32_t ipv4_addr_le = ((ipv4_header->dst_addr >> 24) & 0xff) |
          ((ipv4_header->dst_addr << 8) & 0xff0000) |
          ((ipv4_header->dst_addr >> 8) & 0xff00) |
          ((ipv4_header->dst_addr << 24) & 0xff000000);

  if ((lpm4_table_ptr + (ipv4_addr_le >> 16)) != nullptr) {
    IPv4RuleEntry *entry = (lpm4_table_ptr + (ipv4_addr_le >> 16));
    if (entry->valid_flag) {
      if (entry->next_hop < num_of_eth_devs) {
        EtherCopy(&custom_ether_ip_header->ether_header, port_id, entry->next_hop, dev_mac_addresses_array);
      }
      custom_ether_ip_header->dst_port = entry->next_hop;
    } else {
      custom_ether_ip_header->dst_port = 254;
    }
  }
}

__device__ void IPv6Processing(CustomEtherIPHeader *custom_ether_ip_header, IPv6RuleEntry *lpm6_table_ptr,
                               uint8_t port_id, ether_addr *dev_mac_addresses_array, unsigned int num_of_eth_devs) {
  struct ipv6_hdr *ipv6_header = &custom_ether_ip_header->ipv6_header;

  // Unlike ipv4, we don't have to do edian transfer in ipv6 addr
  uint8_t *ipv6_addr = ipv6_header->dst_addr;

  // Flow of parsing ipv6 headers
  // 1. Check lookup first 24 bits.
  // 2. If the longest matched, and is exactly 24 bits, checkout if there is an external flag.
  // 3. Recursively checkout tbl8 until there is not matched.
  // 4. If matched,
  // 5. Copy the ethernet addresses.
  // 6. Assign the next hop to the custom_ether_ip_header->dst_port.
  // 7. If no rules matched,
  // 8. Assign the custom_ether_ip_header->dst_port = 254

  unsigned long int ipv6_addr_first24 = ipv6_addr[0] << 16 | ipv6_addr[1] << 8 | ipv6_addr[2];
  if ((lpm6_table_ptr + ipv6_addr_first24) != nullptr) {
    IPv6RuleEntry *entry = lpm6_table_ptr + ipv6_addr_first24;
    if (entry->valid_flag) {
      // Match first 24 bits
      if (entry->external_flag && (entry->tbl8_ptr != nullptr)) {
        unsigned int shift_index = 3;
        auto *current = entry->tbl8_ptr;
        auto *match_entry = entry;
        while (shift_index < 16) {
          // Move pointers to indexed rule entry.
          current = current + ipv6_addr[shift_index++];
          if ((current != nullptr) && current->valid_flag) {
            match_entry = current;
            // Match and existed!
            if (current->external_flag && (current->tbl8_ptr != nullptr)) {
              current = current->tbl8_ptr;
            } else {
              // Final match
              if (match_entry->next_hop < num_of_eth_devs) {
                EtherCopy(&custom_ether_ip_header->ether_header, port_id, current->next_hop, dev_mac_addresses_array);
              }
              custom_ether_ip_header->dst_port = match_entry->next_hop;
              break;
            }
          } else {
            // Final match, try_match is
            if (match_entry->next_hop < num_of_eth_devs) {
              EtherCopy(&custom_ether_ip_header->ether_header, port_id, match_entry->next_hop, dev_mac_addresses_array);
            }
            custom_ether_ip_header->dst_port = match_entry->next_hop;
            break;
          }
        }
      } else {
        if (entry->next_hop < num_of_eth_devs) {
          EtherCopy(&custom_ether_ip_header->ether_header, port_id, entry->next_hop, dev_mac_addresses_array);
        }
        custom_ether_ip_header->dst_port = entry->next_hop;
      }
    } else {
      // Not matched, ignore.
      custom_ether_ip_header->dst_port = 254;
    }
  }
}

__global__ void PacketProcessing(CustomEtherIPHeader *dev_custom_ether_ip_header_burst,
                                 uint8_t port_id,
                                 IPv4RuleEntry *lpm4_table_ptr,
                                 IPv6RuleEntry *lpm6_table_ptr,
                                 ether_addr *dev_mac_addresses_array,
                                 unsigned int num_of_eth_devs,
                                 int nb_of_ip_hdrs) {
  int idx = threadIdx.x;
  if (idx < nb_of_ip_hdrs) {
    // Match up packet types.
    if(dev_custom_ether_ip_header_burst[idx].ether_header.ether_type ==
            (((ETHER_TYPE_IPv4 >> 8) | (ETHER_TYPE_IPv4 << 8)) & 0xffff)) {
      // IPv4 header
      IPv4Processing(&dev_custom_ether_ip_header_burst[idx], lpm4_table_ptr, port_id, dev_mac_addresses_array,
                     num_of_eth_devs);
    } else if (dev_custom_ether_ip_header_burst[idx].ether_header.ether_type ==
            (((ETHER_TYPE_IPv6 >> 8) | (ETHER_TYPE_IPv6 << 8)) & 0xffff)) {
      // IPv6 header
      IPv6Processing(&dev_custom_ether_ip_header_burst[idx], lpm6_table_ptr, port_id, dev_mac_addresses_array,
                     num_of_eth_devs);
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

  // Eliminate error checking effort.
  cudaMemcpyAsync(self_batch->dev_custom_ether_ip_headers_burst,
                  self_batch->host_custom_ether_ip_headers_burst,
                  self_batch->nb_rx * sizeof(CustomEtherIPHeader),
                  cudaMemcpyHostToDevice,
                  self_batch->cuda_stream);

  PacketProcessing<<<1, self_batch->nb_rx, 0, self_batch->cuda_stream>>>(self_batch->dev_custom_ether_ip_headers_burst,
          port_id,
          lpm4_table_ptr,
          lpm6_table_ptr,
          dev_mac_addresses_array,
          num_of_eth_devs,
          self_batch->nb_rx);

  cudaMemcpyAsync(self_batch->host_custom_ether_ip_headers_burst,
                  self_batch->dev_custom_ether_ip_headers_burst,
                  self_batch->nb_rx * sizeof(CustomEtherIPHeader),
                  cudaMemcpyDeviceToHost,
                  self_batch->cuda_stream);

  cudaStreamAddCallback(self_batch->cuda_stream, [](cudaStream_t stream, cudaError_t status, void *self_batch_ptr) {
    auto *_self_batch = (ProcessingBatchFrame *) self_batch_ptr;
    _self_batch->ready_to_burst = true;
  }, self_batch, 0);

  return 0;
}

CudaASyncLCoreFunction::CudaASyncLCoreFunction(uint8_t _port_id, unsigned int _num_of_eth_devs,
                                               std::vector<ether_addr> *_mac_addresses_ptr,
                                               IPv4RuleEntry *_lpm4_table_ptr, IPv6RuleEntry *_lpm6_table_ptr)
        : port_id(_port_id), num_of_eth_devs(_num_of_eth_devs), mac_addresses_ptr(_mac_addresses_ptr),
          lpm4_table_ptr(_lpm4_table_ptr), lpm6_table_ptr(_lpm6_table_ptr) {
  // Do nothing
}

} // namespace cu
} // namespace gpuflow

