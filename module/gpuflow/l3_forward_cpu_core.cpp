/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include <rte_ethdev.h>
#include <iostream>
#include <rte_arp.h>
#include "l3_forward_cpu_core.h"

namespace gpuflow {

L3ForwardCPUCore::L3ForwardCPUCore(std::vector<ether_addr> *mac_addresses_ptr) : mac_addresses_ptr(mac_addresses_ptr) {
  // Initialize LPM
  // Create route array
  CreateIPv4LPMRouteArray();
  // FIXME: First, only point at non-numa socket 0
  CreateLPMTable(0);
}

void L3ForwardCPUCore::CreateIPv4LPMRouteArray() {
  for(uint8_t i = 0; i < 4; ++i) {
    // The routing table is located in, ip addr 10.(0-3).x.x, netmask 255.255.0.0
    ipv4_lpm_route_array[i].ip = IPv4(17, i, 0, 0);
    ipv4_lpm_route_array[i].depth = 16;
    ipv4_lpm_route_array[i].if_out = i;
  }
}

void L3ForwardCPUCore::CreateLPMTable(int socket_id) {
  char table_name[64];
  snprintf(table_name, sizeof(table_name), "IPV4_L3FWD_LPM_%d", socket_id);

  rte_lpm_config config_ipv4;
  config_ipv4.max_rules = MAX_LPM_IPV4_RULES;
  config_ipv4.number_tbl8s = (1 << 8);
  config_ipv4.flags = 0;
  ipv4_lpm_lookup_struct[socket_id] = rte_lpm_create(table_name, socket_id, &config_ipv4);
  if (ipv4_lpm_lookup_struct[socket_id] == nullptr) {
    std::cerr << "Can't create ipv4 lpm look up struct on socket id : " << socket_id << std::endl;
    exit(1);
  }
  int ret = 0;
  for(unsigned int i = 0; i < 4; ++i) {
    ret = rte_lpm_add(ipv4_lpm_lookup_struct[socket_id], ipv4_lpm_route_array[i].ip,
                      ipv4_lpm_route_array[i].depth, ipv4_lpm_route_array[i].if_out);
    if (ret < 0) {
      std::cout << rte_strerror(-rte_errno) << std::endl;
      std::cerr << "Error occurred on add lpm, abort" << std::endl;
      exit(1);
    }
  }
}

inline uint16_t L3ForwardCPUCore::LPMLookUp(ipv4_hdr *ipv4_header, uint16_t port_id, int socket_id) {
  unsigned int next_hop;
  return (uint16_t ) ((rte_lpm_lookup(ipv4_lpm_lookup_struct[socket_id],
                                      rte_be_to_cpu_32(ipv4_header->dst_addr),
                                      &next_hop) == 0)? next_hop : port_id);
}

void L3ForwardCPUCore::SimpleLPMForward(rte_mbuf *mbuf, unsigned int port_id, int socket_id){
  ether_hdr *ether_header;
  ipv4_hdr *ipv4_header;
  ether_header = rte_pktmbuf_mtod(mbuf, ether_hdr *);

  // Check out if ipv4 header
  if (RTE_ETH_IS_IPV4_HDR(mbuf->packet_type)) {
    ipv4_header = rte_pktmbuf_mtod_offset(mbuf, ipv4_hdr *, sizeof(ether_hdr));
    uint16_t dst_port = LPMLookUp(ipv4_header, (uint16_t) port_id, socket_id);
    // TODO: Make a more bounded check.
    if (dst_port >= RTE_MAX_ETHPORTS) {
      dst_port = (uint16_t) port_id;
    }
    // Copy the ethernet address a.k.a mac address to the next hop packet.
    ether_header = rte_pktmbuf_mtod(mbuf, ether_hdr *);
    // Set the destination mac addr
    memcpy(&ether_header->d_addr, &mac_addresses_ptr->at(dst_port), sizeof(ether_addr));
    // Set the source mac addr
    memcpy(&ether_header->s_addr, &mac_addresses_ptr->at(port_id), sizeof(ether_addr));
    unsigned int send = rte_eth_tx_burst((uint8_t) dst_port, 0, &mbuf, 1);
    if (send > 0) {
      // Send
      std::cout << "Send from port " << port_id << ", to port " << dst_port << std::endl;
    } else {
      // Clean up non-send
      rte_pktmbuf_free(mbuf);
    }
  } else {
    // Checkout if it's an arp packet, to resolve for arp table.
    ether_header = rte_pktmbuf_mtod(mbuf, ether_hdr *);
    if (ether_header->ether_type == rte_cpu_to_be_16(ETHER_TYPE_ARP)) {
      // Send the packet to all other ports.
      // FIXME: Hardcoded port nums
      for (unsigned int predicates = 0; predicates < 4; ++predicates) {
        if (predicates == port_id) {
          // Don't send to self, continue
        } else {
          unsigned int send = rte_eth_tx_burst((uint8_t) predicates, 0, &mbuf, 1);
          if (send > 0) {
            // Success, ignore.
          } else {
            rte_pktmbuf_free(mbuf);
          }
        }
      }
    }
    // Drop the non-ipv4 packet.
    rte_pktmbuf_free(mbuf);
  }
}

void L3ForwardCPUCore::LCoreFunctions() {
  unsigned int lcore_id;
  int ret;
  RTE_LCORE_FOREACH_SLAVE(lcore_id) {
    ret = rte_eal_remote_launch([](void *arg) -> int {
      unsigned int self_lcore_id = rte_lcore_id();
      unsigned int port_id = (self_lcore_id > 0) ? self_lcore_id -1 : self_lcore_id;
      auto *self = (L3ForwardCPUCore *)arg;
      while(true) {
        struct rte_mbuf *pkts_burst[32];
        // Receive
        const unsigned int nb_rx = rte_eth_rx_burst(port_id, 0, pkts_burst, 32);
        for (unsigned int idx = 0; idx < nb_rx; ++idx) {
          struct rte_mbuf *mbuf = pkts_burst[idx];
          // Forward in lpm manner.
          self->SimpleLPMForward(mbuf, port_id, 0);
        }
      }
    }, (void *)this, lcore_id);

    if (ret < 0) {
      std::cerr << "Error occurred on executing DumpPacketCore LCoreFunctions, abort" << std::endl;
      exit(1);
    }

  }
  rte_eal_mp_wait_lcore();
}

} // namespace gpuflow