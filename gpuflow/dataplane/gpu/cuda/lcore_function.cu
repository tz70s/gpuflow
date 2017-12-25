// CUDA source

#include <stdio.h>
#include <rte_mbuf.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "lcore_function.h"

__global__ void advanceParticles() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
}

namespace gpuflow {

void CheckoutDevices() {
	int device_ids;
	printf("Hello world\n");
}

} // namespace gpuflow

