/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include <rte_ethdev.h>
#include <iostream>
#include "l3_forward_gpu_core.h"
#include "cuda/lcore_function.h"

namespace gpuflow {

L3ForwardGPUCore::L3ForwardGPUCore(unsigned int num_of_eth_devs, std::vector<ether_addr> *mac_addresses_ptr) 
				: DataPlaneCore(num_of_eth_devs), mac_addresses_ptr(mac_addresses_ptr) {
	// Nothing to do, currently.
}

void L3ForwardGPUCore::LCoreFunctions() {
	CheckoutDevices();
}

} // namespace gpuflow
