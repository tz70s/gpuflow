/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include <rte_lcore.h>
#include <iostream>
#include <rte_mbuf.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <rte_ip.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include "dataplane_core.h"

namespace gpuflow {

void SayHelloCore::LCoreFunctions() {
  // Launch lcore for processing
  unsigned int lcore_id;
  int ret;
  RTE_LCORE_FOREACH_SLAVE(lcore_id) {
    ret = rte_eal_remote_launch([](void *) -> int {
            unsigned int self_lcore_id = rte_lcore_id();
            std::cout << "Hi, I'm " << self_lcore_id << " lcore" << std::endl;
            return 0;
    }, nullptr, lcore_id);
    if (ret < 0) {
      std::cerr << "Error occured, in ret " << ret << std::endl;
      exit(1);
    }
  }
  rte_eal_mp_wait_lcore();
}

void DumpPacketCore::LCoreFunctions() {
  unsigned int lcore_id;
  int ret;

  RTE_LCORE_FOREACH_SLAVE(lcore_id) {
    ret = rte_eal_remote_launch([](void *arg) -> int {
      unsigned int self_lcore_id = rte_lcore_id();
      unsigned int port_id = (self_lcore_id > 0) ? self_lcore_id -1 : self_lcore_id;
      auto *self = (DumpPacketCore *)arg;
      int count = 0;
      while(true) {
        struct rte_mbuf *pkts_burst[32];
        const unsigned int nb_rx = rte_eth_rx_burst(port_id, 0, pkts_burst, 32);
        for (unsigned int i = 0; i < nb_rx; ++i) {
          struct rte_mbuf *mbuf = pkts_burst[i];
          std::cout << "port : " << port_id << ", rx : " << count++ << std::endl;
        }
      }
    }, (void *)this, lcore_id);
    if (ret < 0) {
      std::cerr << "Error occured on executing DumpPacketCore LCoreFunctions, abort" << std::endl;
      exit(1);
    }
  }
  rte_eal_mp_wait_lcore();
}

void BasicForwardCore::LCoreFunctions() {
  unsigned int lcore_id;
  int ret;
  RTE_LCORE_FOREACH_SLAVE(lcore_id) {
    ret = rte_eal_remote_launch([](void *arg) -> int {
      unsigned int self_lcore_id = rte_lcore_id();
      unsigned int port_id = (self_lcore_id > 0) ? self_lcore_id -1 : self_lcore_id;
      auto *self = (BasicForwardCore *)arg;
      int count = 0;
      while(true) {
        struct rte_mbuf *pkts_burst[32];
        // Receive
        const unsigned int nb_rx = rte_eth_rx_burst(port_id, 0, pkts_burst, 32);
        for (unsigned int idx = 0; idx < nb_rx; ++idx) {
          struct rte_mbuf *mbuf = pkts_burst[idx];
          struct ether_hdr *ether = rte_pktmbuf_mtod(mbuf, ether_hdr *);
          unsigned int send = rte_eth_tx_burst(port_id ^ 3, 0, &mbuf, 1);
          if (send) {
            std::cout << "Transfer a packet! From dtap" << port_id << " to dtap" << (port_id ^ 3) << std::endl;
          } else {
            // clean up
            rte_pktmbuf_free(mbuf);
          }
        }
      }
    }, (void *)this, lcore_id);
    if (ret < 0) {
      std::cerr << "Error occured on executing DumpPacketCore LCoreFunctions, abort" << std::endl;
      exit(1);
    }
  }
  rte_eal_mp_wait_lcore();
}

} // namespace gpuflow
