/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include <rte_ethdev.h>
#include <iostream>
#include "l3_forward_gpu_core.h"
#include "dataplane/gpu/cuda/cuda_async_lcore_function.h"

namespace gpuflow {

L3ForwardGPUCore::L3ForwardGPUCore(unsigned int num_of_eth_devs, std::vector<ether_addr> *mac_addresses_ptr)
        : DataPlaneCore(num_of_eth_devs), mac_addresses_ptr(mac_addresses_ptr) {
  // Nothing to do, currently.
}

void L3ForwardGPUCore::LCoreFunctions() {
  unsigned int lcore_id;
  int ret;

  RTE_LCORE_FOREACH_SLAVE(lcore_id) {
    ret = rte_eal_remote_launch([](void *arg) -> int {
      unsigned int self_lcore_id = rte_lcore_id();
      unsigned int port_id = (self_lcore_id > 0) ? self_lcore_id -1 : self_lcore_id;
      auto *self = (L3ForwardGPUCore *)arg;
      if (port_id == 0) {
        cu::CudaASyncLCoreFunction cuda_lcore_function(self->num_of_eth_devs, self->mac_addresses_ptr);

        while (true) {
          struct rte_mbuf *pkts_burst[32];
          // Receive
          const unsigned int nb_rx = rte_eth_rx_burst(port_id, 0, pkts_burst, 32);
          // Call out cuda function
          if (nb_rx > 0) {
            cuda_lcore_function.ProcessPacketsBatch(pkts_burst, nb_rx);
          }
        }
      }
      return 0;
    }, (void *)this, lcore_id);

    if (ret < 0) {
      std::cerr << "Error occurred on executing DumpPacketCore LCoreFunctions, abort" << std::endl;
      exit(1);
    }

  }
  rte_eal_mp_wait_lcore();
}

} // namespace gpuflow
