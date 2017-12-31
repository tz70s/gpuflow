/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include "dataplane/dataplane_core.h"
#include "dataplane/dataplane.h"
#include "dataplane/gpu/l3_forward_gpu_core.h"
#include "dataplane/gpu/cuda/cuda_async_lcore_function.h"
#include <iostream>
#include <cuda_profiler_api.h>
#include <signal.h>

#ifdef _GPU_EXEC
#define GPUFLOW_EXECUTION gpuflow::L3ForwardGPUCore_t
#else
#define GPUFLOW_EXECUTION gpuflow::L3ForwardCPUCore_t
#endif

void SignalHandler(int sig_num) {
  std::cout << "\nExit program via user interrupt " << std::endl;
  cudaProfilerStop();
  exit(0);
}

int main(int argc, char *argv[]) {
  cudaProfilerStart();
  signal(SIGINT, SignalHandler);
  // Create a gpu accelerated data plane
  gpuflow::DataPlane data_plane(argc, argv);
  data_plane.ServeProcessingLoop(GPUFLOW_EXECUTION);
  return 0;
}
