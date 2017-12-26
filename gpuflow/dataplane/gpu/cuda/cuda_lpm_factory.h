/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _DATAPLANE_LPM_CU_H_
#define _DATAPLANE_LPM_CU_H_

#include <cstdint>
#include <cuda_runtime_api.h>

namespace gpuflow {

struct IPv4RuleEntry {
  uint8_t next_hop;
  bool valid_flag;
  // The external flag will be used when the depth exceed 24 bits.
  bool external_flag;
  uint8_t depth;
};

namespace cu {

class IPv4LPMFactory {
 public:
  IPv4LPMFactory() : IPv4TBL24(nullptr) {
    // 2 ^ 24 rules
    MAX_LPM_ROUTING_RULES = 1;
    for (int i = 0; i < 16; i++) {
      MAX_LPM_ROUTING_RULES *= 2;
    }
  }
  unsigned const int MAX_DEPTH = 24;
  unsigned long MAX_LPM_ROUTING_RULES;
  // We'll have num of MAX_LPM_ROUTING_RULES IPv4RuleEntry.
  // This pointer will be accessed via the device (GPU) side.
  IPv4RuleEntry *IPv4TBL24;
  int CreateLPMTable();
  int AddLPMRule(uint32_t ipv4_address, uint8_t depth, uint8_t next_hop);
  ~IPv4LPMFactory() {
    cudaFree(IPv4TBL24);
  }
};

} // namespace cu
} // namespace gpuflow


#endif // _DATAPLANE_LPM_CU_H_
