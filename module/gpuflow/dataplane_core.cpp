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
      while(true) {
        struct rte_mbuf *pkts_burst[32];
        const unsigned int nb_rx = rte_eth_rx_burst(port_id, 0, pkts_burst, 32);
        for (unsigned int i = 0; i < nb_rx; ++i) {
          struct rte_mbuf *mbuf = pkts_burst[i];
          std::cout << mbuf->pkt_len << std::endl;
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
      while(true) {
        struct rte_mbuf *pkts_burst[32];
        const unsigned int nb_rx = rte_eth_rx_burst(port_id, 0, pkts_burst, 32);
        for (unsigned int i = 0; i < nb_rx; ++i) {
          struct rte_mbuf *mbuf = pkts_burst[i];
          std::cout << mbuf->pkt_len << std::endl;
          const uint16_t nb_tx = rte_eth_tx_burst(1, 0, &mbuf, nb_rx);
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
