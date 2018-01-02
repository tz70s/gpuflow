/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include <rte_ethdev.h>
#include <iostream>
#include "l3_forward_gpu_core.h"
#include "dataplane/gpu/cuda/cuda_async_lcore_function.h"
#include <chrono>

namespace gpuflow {

L3ForwardGPUCore::L3ForwardGPUCore(unsigned int num_of_eth_devs, std::vector<ether_addr> *mac_addresses_ptr)
        : DataPlaneCore(num_of_eth_devs), mac_addresses_ptr(mac_addresses_ptr) {
  // Nothing to do, currently.
}

void L3ForwardGPUCore::SendOut(cu::ProcessingBatchFrame *batch_ptr, uint8_t self_port) {
  for (int i = 0; i < batch_ptr->nb_rx; ++i) {
    struct rte_mbuf *mbuf = batch_ptr->pkts_burst[i];
    struct ether_hdr *ether = rte_pktmbuf_mtod(mbuf, struct ether_hdr *);
    rte_memcpy(ether, &batch_ptr->host_custom_ether_ip_headers_burst[i], sizeof(struct ether_hdr));
    if (batch_ptr->host_custom_ether_ip_headers_burst[i].dst_port == (uint8_t) 255) {
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
      if (batch_ptr->host_custom_ether_ip_headers_burst[i].dst_port > (uint8_t) num_of_eth_devs) {
        // Drop out, non configured port.
      } else {
        int send = rte_eth_tx_burst(batch_ptr->host_custom_ether_ip_headers_burst[i].dst_port, 0, &mbuf, 1);
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
  RTE_LCORE_FOREACH_SLAVE(lcore_id) {
    ret = rte_eal_remote_launch([](void *arg) -> int {
      unsigned int self_lcore_id = rte_lcore_id();
      unsigned int port_id = (self_lcore_id > 0) ? self_lcore_id -1 : self_lcore_id;
      auto *self = (L3ForwardGPUCore *) arg;
      cu::CudaASyncLCoreFunction cuda_lcore_function(port_id, self->num_of_eth_devs, self->mac_addresses_ptr,
                                                     self->data_plane_lpm_ipv4_gpu.IPv4TBL24);
      cuda_lcore_function.SetupCudaDevices();
      int num_of_batches = 16;
      cuda_lcore_function.CreateProcessingBatchFrame(num_of_batches, 128);
      int batch_index = 0;
      // Start receive time
      auto base_clock = std::chrono::high_resolution_clock::now();
      while (true) {
        // At first, we'll drain the current waiting for transferring from all batches.
        // Due to iterate all the batches, that is, the num of batches should be carefully match.
        for (int i = 0; i < num_of_batches; i++) {
          auto *local_batch = cuda_lcore_function.batch_head[i];
          // Checkout if batch complete the gpu operations.
          // We'll check if the batch is busy, and ready to burst out.
          if (local_batch->busy && local_batch->ready_to_burst) {
            // Burst
            self->SendOut(local_batch, port_id);
            // Set to not busy and not ready to burst
            local_batch->busy = false;
            local_batch->ready_to_burst = false;
            local_batch->nb_rx = 0;
          }
        }

        bool clock_due = false;
        // base line clock time.
        base_clock = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_of_batches; i++) {
          // While if we check the current transferring, we'll going to accept new packet.
          // Retrieve a free batch.
          auto *current_batch = cuda_lcore_function.batch_head[batch_index];
          if (current_batch->busy) {
            batch_index++;
            if (batch_index == num_of_batches) {
              batch_index = 0;
            }
            // Go back to top, for iterate to next batch.
            continue;
          }
          while (true) {
            // Receive num of packets.
            // The max of 2 receive will be exactly to maximum of batch sizes to avoid overflow.
            const unsigned int nb_rx = rte_eth_rx_burst(port_id, 0, &current_batch->pkts_burst[current_batch->nb_rx],
                                                        64);
            // Checkout if there's any packet.
            if (nb_rx > 0) {
              current_batch->busy = true;
              // Copy data to pinned host memory
              for (uint8_t i = 0; i < nb_rx; ++i) {
                rte_memcpy(&current_batch->host_custom_ether_ip_headers_burst[current_batch->nb_rx + i],
                           rte_pktmbuf_mtod(current_batch->pkts_burst[current_batch->nb_rx + i], struct ether_hdr*),
                           sizeof(cu::CustomEtherIPHeader) - sizeof(uint8_t));
              }
              current_batch->nb_rx += nb_rx;
              if (current_batch->nb_rx > 64) {
                cuda_lcore_function.ProcessPacketsBatch(current_batch);
                break;
              }
            }
            auto current_clock = std::chrono::high_resolution_clock::now();
            // Interval 200us
            if (std::chrono::duration_cast<std::chrono::microseconds>(current_clock - base_clock).count() > 200) {
              if (current_batch->nb_rx > 0) {
                cuda_lcore_function.ProcessPacketsBatch(current_batch);
              }
              clock_due = true;
              break;
            }
          }
          batch_index++;
          if (batch_index == num_of_batches) {
            batch_index = 0;
          }
          if (clock_due) break;
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
