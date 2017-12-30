/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include <rte_ethdev.h>
#include <iostream>
#include "l3_forward_gpu_core.h"
#include "dataplane/gpu/cuda/cuda_async_lcore_function.h"
#include <cuda_profiler_api.h>
#include <signal.h>

namespace gpuflow {

L3ForwardGPUCore::L3ForwardGPUCore(unsigned int num_of_eth_devs, std::vector<ether_addr> *mac_addresses_ptr)
        : DataPlaneCore(num_of_eth_devs), mac_addresses_ptr(mac_addresses_ptr) {
  // Nothing to do, currently.
}

void SignalHandler(int sig_num) {
  std::cout << "\nExit program via user interrupt " << std::endl;
  cudaProfilerStop();
  exit(0);
}

void L3ForwardGPUCore::SendOut(cu::ProcessingBatchFrame *batch_ptr, uint8_t self_port) {
  for (int i = 0; i < batch_ptr->nb_rx; ++i) {
    struct rte_mbuf *mbuf = batch_ptr->pkts_burst[i];
    struct ether_hdr *ether = rte_pktmbuf_mtod(mbuf, struct ether_hdr *);
    rte_memcpy(ether, &batch_ptr->host_custom_ether_ip_headers_burst[i], sizeof(struct ether_hdr));
    if (batch_ptr->host_dst_ports_burst[i] == (uint8_t) 255) {
      // Broadcast
      for (uint8_t port = 0; port < num_of_eth_devs; port++) {
        if (port == self_port) {
          continue;
        }
        int send = rte_eth_tx_burst(port, 0, &mbuf, 1);
        if (send > 0) {
          // success
        } else {
          // drop
          rte_pktmbuf_free(mbuf);
        }
      }
    } else {
      if (batch_ptr->host_dst_ports_burst[i] > (uint8_t) num_of_eth_devs) {
        // Drop out, non configured port.
      } else {
        int send = rte_eth_tx_burst(batch_ptr->host_dst_ports_burst[i], 0, &mbuf, 1);
        if (send > 0) {
          // success
        } else {
          // drop
          rte_pktmbuf_free(mbuf);
        }
      }
    }
  }
}

void L3ForwardGPUCore::LCoreFunctions() {
  unsigned int lcore_id;
  int ret;
  cudaProfilerStart();

  RTE_LCORE_FOREACH_SLAVE(lcore_id) {
    ret = rte_eal_remote_launch([](void *arg) -> int {
      unsigned int self_lcore_id = rte_lcore_id();
      unsigned int port_id = (self_lcore_id > 0) ? self_lcore_id -1 : self_lcore_id;
      auto *self = (L3ForwardGPUCore *) arg;
      cu::CudaASyncLCoreFunction cuda_lcore_function(port_id, self->num_of_eth_devs, self->mac_addresses_ptr,
                                                     self->data_plane_lpm_ipv4_gpu.IPv4TBL24);
      cuda_lcore_function.SetupCudaDevices();
      int num_of_batches = 10;
      cuda_lcore_function.CreateProcessingBatchFrame(num_of_batches, 32);
      int batch_index = 0;
      while (true) {
        // Drain tx
        for (int i = 0; i < num_of_batches; i++) {
          // Checkout if batch complete the gpu operations
          auto *local_batch = cuda_lcore_function.batch_head[i];
          if (local_batch->busy && local_batch->ready_to_burst) {
            // Burst
            self->SendOut(local_batch, port_id);
            // Set to not busy and not ready to burst
            local_batch->busy = false;
            local_batch->ready_to_burst = false;
          }
        }
        auto *current_batch = cuda_lcore_function.batch_head[batch_index];
        struct rte_mbuf *pkts_burst[32];
        // Receive
        const unsigned int nb_rx = rte_eth_rx_burst(port_id, 0, pkts_burst, 32);
        // Call out cuda function
        if (nb_rx > 0) {
          current_batch->pkts_burst = pkts_burst;
          if (current_batch->busy == true) {
            std::cout << "Retrieve a busy batch, iterate to next." << std::endl;
            batch_index++;
            if (batch_index == 10) {
              batch_index = 0;
            }
            // Iterate to next batch
            continue;
          } else {
            current_batch->busy = true;
            // Copy data to pinned host memory
            for (uint8_t i = 0; i < nb_rx; ++i) {
              rte_memcpy(&current_batch->host_custom_ether_ip_headers_burst[i],
                         rte_pktmbuf_mtod(pkts_burst[i], struct ether_hdr*),
                         sizeof(cu::CustomEtherIPHeader));
            }
            current_batch->nb_rx = nb_rx;
            // TODO: Pass pointer
            cuda_lcore_function.ProcessPacketsBatch(batch_index++, nb_rx);
            if (batch_index == 10) {
              batch_index = 0;
            }
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

  signal(SIGINT, SignalHandler);

  rte_eal_mp_wait_lcore();
}

} // namespace gpuflow
