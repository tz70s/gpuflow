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
#include "dataplane_processor.h"

namespace gpuflow {

void SayHelloProcessor::LCoreFunctions() {
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

void DumpPacketProcessor::LCoreFunctions() {
  unsigned int lcore_id;
  int ret;

  RTE_LCORE_FOREACH_SLAVE(lcore_id) {
    ret = rte_eal_remote_launch([](void *arg) -> int {
      unsigned int self_lcore_id = rte_lcore_id();
      DumpPacketProcessor *self = (DumpPacketProcessor *)arg;
      while(true) {
        int ret = 0;
        rte_mbuf *mbuf = rte_pktmbuf_alloc(self->pkt_mbuf_pool);
        if (mbuf == nullptr) continue;
        ret = read(self->tap_fds->at(0), rte_pktmbuf_mtod(mbuf, struct ether_hdr *), 2048);
        if (ret < 0) {
          std::cerr << "Can't read from tap_fd" << std::endl;
          exit(1);
        }
        // FIXME : Can't work on reading icmp frame
        std::cout << "Read : " << rte_pktmbuf_data_len(mbuf) << ", ret: " << ret << std::endl;
        std::cout << (char *)(mbuf->userdata) << std::endl;
        mbuf->nb_segs = 1;
        mbuf->next = nullptr;
        mbuf->pkt_len = (uint16_t)ret;
        mbuf->data_len = (uint16_t)ret;
      }
    }, (void *)this, lcore_id);
    if (ret < 0) {
      std::cerr << "Error occured on executing DumpPacketProcessor LCoreFunctions, abort" << std::endl;
      exit(1);
    }
  }
  rte_eal_mp_wait_lcore();
}

} // namespace gpuflow
