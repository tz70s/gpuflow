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
                               uint8_t port_id, ether_addr *dev_mac_addresses_array, uint8_t *dst_port) {

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
      *dst_port = entry->next_hop;
#ifdef _DEBUG
      printf("Received IPv4 packet, next hop -> %d\n", entry->next_hop);
#endif
    } else {
      *dst_port = 254;
    }
  }
}

__device__ void IPv6Processing(CustomEtherIPHeader *custom_ether_ip_header) {
  // TODO : Add ipv6 processing
}

__global__ void PacketProcessing(CustomEtherIPHeader *dev_custom_ether_ip_header_burst,
                                 uint8_t port_id,
                                 uint8_t *dev_dst_port_burst,
                                 IPv4RuleEntry *lpm_table_ptr,
                                 ether_addr *dev_mac_addresses_array,
                                 int nb_of_ip_hdrs) {
  int idx = threadIdx.x;
  if (idx < nb_of_ip_hdrs) {
    // Match up packet types.
    if(dev_custom_ether_ip_header_burst[idx].ether_header.ether_type ==
            (((ETHER_TYPE_IPv4 >> 8) | (ETHER_TYPE_IPv4 << 8)) & 0xffff)) {
      // IPv4 header
      IPv4Processing(&dev_custom_ether_ip_header_burst[idx], lpm_table_ptr, port_id, dev_mac_addresses_array,
                     &dev_dst_port_burst[idx]);
    } else if (dev_custom_ether_ip_header_burst[idx].ether_header.ether_type ==
            (((ETHER_TYPE_IPv6 >> 8) | (ETHER_TYPE_IPv6 << 8)) & 0xffff)) {
      // IPv6 header
      IPv6Processing(&dev_custom_ether_ip_header_burst[idx]);
      dev_dst_port_burst[idx] = 254;
    } else if (dev_custom_ether_ip_header_burst[idx].ether_header.ether_type ==
            (((ETHER_TYPE_ARP >> 8) | (ETHER_TYPE_ARP << 8)) & 0xffff)){
      // Send to all
      dev_dst_port_burst[idx] = 255;
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

ProcessingBatchFrame::ProcessingBatchFrame(uint8_t _batch_size) : pkts_burst(nullptr), batch_size(_batch_size),
                                                                  busy(false) {
  for (int i = 0; i < 32; i++) {
    cudaStreamCreate(&cuda_stream[i]);
  }
  CudaMallocWithFailOver((void **) &dev_custom_ether_ip_headers_burst, batch_size * sizeof(CustomEtherIPHeader),
                         "dev_custom_ether_ip_headers_burst");
  CudaMallocWithFailOver((void **) &dev_dst_ports_burst, batch_size * sizeof(uint8_t), "dev_dst_ports_burst");
}

void CudaASyncLCoreFunction::CreateProcessingBatchFrame(int num_of_batch, uint8_t batch_size){
  batch_head = new ProcessingBatchFrame *[num_of_batch];
  for (int i = 0; i < num_of_batch; i++) {
    batch_head[i] = new ProcessingBatchFrame(batch_size);
  }
}

int CudaASyncLCoreFunction::ProcessPacketsBatch(int batch_idx, struct rte_mbuf **pkts_burst, int nb_rx) {
  auto self_batch = batch_head[batch_idx];
  self_batch->pkts_burst = pkts_burst;
  if (self_batch->busy) {
    std::cout << "Retrieve a busy batch, error occurred, abort." << std::endl;
    return -1;
  }
  self_batch->busy = true;
  for (uint8_t i = 0; i < nb_rx; ++i) {
    CudaASyncMemcpyWithFailOver(&self_batch->dev_custom_ether_ip_headers_burst[i],
                                rte_pktmbuf_mtod(pkts_burst[i], struct ether_hdr *),
                                sizeof(CustomEtherIPHeader),
                                cudaMemcpyHostToDevice,
                                self_batch->cuda_stream[i],
                                "custom_ether_ip_header_memory_copy");
  }

  for (uint8_t i = 0; i < nb_rx; ++i) {
    PacketProcessing <<< 1, 1, 0, self_batch->cuda_stream[i]>>>(&self_batch->dev_custom_ether_ip_headers_burst[i],
            port_id,
            &self_batch->dev_dst_ports_burst[i],
            lpm_table_ptr,
            dev_mac_addresses_array,
            nb_rx);
  }

  for (uint8_t i = 0; i < nb_rx; ++i) {
    CudaASyncMemcpyWithFailOver(&self_batch->host_dst_ports_burst[i], &self_batch->dev_dst_ports_burst[i],
                                sizeof(uint8_t),
                                cudaMemcpyDeviceToHost, self_batch->cuda_stream[i], "dev_dst_ports_burst_memory_copy_back");
  }

  for (uint8_t index = 0; index < nb_rx; index++) {
    CudaASyncMemcpyWithFailOver(rte_pktmbuf_mtod(self_batch->pkts_burst[index], struct ether_hdr *),
                                &self_batch->dev_custom_ether_ip_headers_burst[index],
                                sizeof(ether_hdr),
                                cudaMemcpyDeviceToHost,
                                self_batch->cuda_stream[index],
                                "custom_ether_header_memory_copy_back");
  }

  // FIXME: Currently, sync here.
  for (int i = 0; i < nb_rx; ++i) {
    cudaStreamSynchronize(self_batch->cuda_stream[i]);
  }
  cudaDeviceSynchronize();

  for (uint8_t i = 0; i < (uint8_t) nb_rx; i++) {
    struct rte_mbuf *mbuf = self_batch->pkts_burst[i];
    if (self_batch->host_dst_ports_burst[i] == (uint8_t) 255) {
      // Broadcast
      for (uint8_t port = 0; port < num_of_eth_devs; port++) {
        if (port == port_id) {
          continue;
        }
        int send = rte_eth_tx_burst(port, 0, &mbuf, 1);
        if (send > 0) {
          // success
        } else {
          // The drop can't be memory aligned in cuda object.
          // We need to drop at cpp file.
          // Although, it's not that necessary to drop it.
        }
      }
    } else {
      if (self_batch->host_dst_ports_burst[i] > (uint8_t) num_of_eth_devs) {
        // Drop out, non configured port.
        continue;
      }
      int send = rte_eth_tx_burst(self_batch->host_dst_ports_burst[i], 0, &mbuf, 1);
      if (send > 0) {
        // success
      } else {
        // drop
      }
    }
  }
  self_batch->busy = false;
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

