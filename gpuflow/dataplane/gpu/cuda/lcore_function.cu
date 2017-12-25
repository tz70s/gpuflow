/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include <stdio.h>
#include <rte_mbuf.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "lcore_function.h"

namespace gpuflow {
namespace cuda {

__global__ void PacketProcess() {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
}

void CheckoutDevices() {
  int device_ids;
  printf("Hello world\n");
}

} // namespace cuda
} // namespace gpuflow

