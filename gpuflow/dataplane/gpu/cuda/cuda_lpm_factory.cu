/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include "cuda_lpm_factory.h"
#include <rte_ip.h>
#include <iostream>
#include <driver_types.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

namespace gpuflow {
namespace cu {

static inline void CudaMallocWithFailOver(void **predicate, size_t size, const char *predicate_type) {
  if (cudaMalloc(predicate, size) != cudaSuccess) {
    std::cerr << "Device memory allocation on " << predicate_type << " failed, abort." << std::endl;
    exit(1);
  }
}

__global__ void InitLPMTable(IPv4RuleEntry *ipv4_tbl_24) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ipv4_tbl_24[idx].next_hop = 254;
  ipv4_tbl_24[idx].valid_flag = false;
  ipv4_tbl_24[idx].depth = 0;
  ipv4_tbl_24[idx].external_flag = false;
}

__global__ void InitIPv6LPMTable(IPv6RuleEntry *ipv6_tbl_24) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ipv6_tbl_24[idx].next_hop = 254;
  ipv6_tbl_24[idx].valid_flag = false;
  ipv6_tbl_24[idx].depth = 0;
  ipv6_tbl_24[idx].external_flag = false;
  ipv6_tbl_24[idx].tbl8_ptr = nullptr;
}

// Create LPM Table
int IPv4LPMFactory::CreateLPMTable() {
  // Allocate lpm table sizes
  CudaMallocWithFailOver((void **)&IPv4TBL24, MAX_LPM_ROUTING_RULES * sizeof(IPv4RuleEntry), "IPv4TBL24");
  unsigned long num_of_threads = 2048;
  InitLPMTable<<<MAX_LPM_ROUTING_RULES/num_of_threads, num_of_threads>>>(IPv4TBL24);
  cudaDeviceSynchronize();
  std::cout << "Initialized lpm entries" << std::endl;
  return 0;
}

// Create IPv6 LPM Table
int IPv6LPMFactory::CreateLPMTable() {
  // Allocate ipv6 lpm table size
  CudaMallocWithFailOver((void **)&IPv6TBL24, MAX_LPM_ROUTING_RULES * sizeof(IPv6RuleEntry), "IPv6TBL24");
  unsigned long num_of_threads = 2048;
  InitIPv6LPMTable<<<MAX_LPM_ROUTING_RULES/num_of_threads, num_of_threads>>>(IPv6TBL24);
  cudaDeviceSynchronize();
  std::cout << "Initialized ipv6 lpm entries" << std::endl;
  return 0;
};

__global__ void SetupRuleEntry(IPv4RuleEntry *ipv4_tbl_24, unsigned long int start, uint8_t next_hop, uint8_t depth) {
  int idx = threadIdx.x;
  ipv4_tbl_24[start + idx].next_hop = next_hop;
  ipv4_tbl_24[start + idx].valid_flag = true;
  ipv4_tbl_24[start + idx].external_flag = false;
  ipv4_tbl_24[start + idx].depth = depth;
  printf("Setup lpm rule entry! index : %lu, next_hop : %d\n", (start + idx), next_hop);
}

int IPv4LPMFactory::AddLPMRule(uint32_t ipv4_address, uint8_t depth, uint8_t next_hop) {
  if (depth > MAX_DEPTH) {
    std::cerr << "Currently, we are not support the tbl 8 secondary search" << std::endl;
    exit(1);
  }

  // FIXME: Back to 24 mask
  unsigned long int start = ipv4_address >> 16 ;
  unsigned long int end = (ipv4_address >> 16) + 1;

  SetupRuleEntry<<<1, end-start>>>(IPv4TBL24, start, next_hop, depth);
  cudaDeviceSynchronize();
  return 0;
}

__global__ void SetupIPv6RuleEntry(IPv6RuleEntry *ipv6_tbl_24, unsigned long int start, uint8_t next_hop, uint8_t depth) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ipv6_tbl_24[start + idx].valid_flag && (ipv6_tbl_24[start + idx].depth > depth)) {
    // There's an existed rule and longer, abort this update
  } else {
    // Add the new rule
    // TODO: Currently, we only handle with the case of depth <= 24. The case of depth > 24 using the tbl8 table has not been implemented.
    ipv6_tbl_24[start + idx].next_hop = next_hop;
    ipv6_tbl_24[start + idx].valid_flag = true;
    ipv6_tbl_24[start + idx].external_flag = false;
    ipv6_tbl_24[start + idx].depth = depth;
    ipv6_tbl_24[start + idx].tbl8_ptr = nullptr;
    printf("Setup ipv6 lpm rule entry! index: %lu, next_hop: %d\n", (start + idx), next_hop);
  }
}

int IPv6LPMFactory::AddLPMRule(uint8_t *ipv6_address, uint8_t depth, uint8_t next_hop) {
  if(depth > MAX_DEPTH) {
    std::cerr << "The depth can't be longer than 128" << std::endl;
    exit(1);
  }

  // Calculate the distance
  unsigned long int distance = 1;
  for (unsigned long int i = 0; i < (24 - depth); i++) {
    distance *= 2;
  }

  unsigned long int start;
  if (depth == 24) {
    // The depth is exactly 24

    // Calculate the start
    start = ipv6_address[0] << 16 | ipv6_address[1] << 8 | ipv6_address[2];
    if ((start + distance) >= (1 << 24)) {
      // overflow
      distance = (1 << 24) - start;
    }

  } else if (depth < 24 && depth >= 16) {
    // The depth is in the range of 16-23.

    // Calculate the  start
    unsigned long int right_shift = ipv6_address[2] >> (24 - depth); 
    start = ipv6_address[0] << 16 | ipv6_address[1] << 8 | right_shift << (24 - depth);

  } else if (depth < 16 && depth >= 8) {
    // The depth is in the range of 8-15

    // Calculate the start
    unsigned long int right_shift = ipv6_address[1] >> (16 - depth);
    start = ipv6_address[0] << 16 | right_shift << (24 - depth);

  } else if (depth < 8) {
    // The depth is in the range of 0-7

    // Calculate the start
    unsigned long int right_shift = ipv6_address[0] >> (8 -depth);
    start = right_shift << (24 - depth);
      
  } else {
    // TODO: Handling when depth is larger than 24
    start = ipv6_address[0] << 16 | ipv6_address[1] << 8 | ipv6_address[2];

  }

  unsigned long num_of_threads = 1024;
  if (distance <= num_of_threads) {
    SetupIPv6RuleEntry<<<1, distance>>>(IPv6TBL24, start, next_hop, depth); 
  } else {
    SetupIPv6RuleEntry<<<distance/num_of_threads, num_of_threads>>>(IPv6TBL24, start, next_hop, depth);
  }
  cudaDeviceSynchronize();
  return 0;
}

} // namespace cu
} // namespace gpuflow
