/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _DATAPLANE_LPM_GPU_H_
#define _DATAPLANE_LPM_GPU_H_

#include <dataplane/gpu/cuda/cuda_lpm_factory.h>

namespace gpuflow {
class DataPlaneLPMIPv4GPU {
 public:
  // FIXED rules in LPM
  DataPlaneLPMIPv4GPU();

  // Delegate pointer on this class.
  IPv4RuleEntry *IPv4TBL24;

 private:
  cu::IPv4LPMFactory ipv4_lpm_factory;
  int CreateLPMTable();
};

} // namespace gpuflow

#endif // _DATAPLANE_LPM_GPU_H_
