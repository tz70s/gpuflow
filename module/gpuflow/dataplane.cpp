/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include "dataplane.h"

#include <iostream>
#include <rte_mempool.h>
#include "l3_forward_cpu_core.h"

namespace gpuflow {

DataPlane::DataPlane(int argc, char *argv[], unsigned int num_of_cores) :
        num_of_cores(num_of_cores),
        pkt_mbuf_pool(nullptr) {
  // Initialize eal
  int ret = rte_eal_init(argc, argv);
  if (ret < 0) {
    rte_exit(EXIT_FAILURE, "ERROR with EAL initialization\n");
  }
  // Find binding eth devs
  int num_of_eth_devs = rte_eth_dev_count();
  if (num_of_eth_devs <= 0) {
    std::cerr << "Didn't find any eth devices, abort" << std::endl;
    exit(1);
  }
  // Create and initialize memory buffer pool
  if (CreateMbufPool() < 0) {
    std::cerr << "Error occurred on creating memory buffer pool of dpdk, abort" << std::endl;
    exit(1);
  }

  InitializePortConf();

  for (int eth_dev_id = 0; eth_dev_id < num_of_eth_devs; ++eth_dev_id) {
    // Initialize ports
    if (InitializePorts((uint8_t) eth_dev_id) < 0) {
      std::cerr << "Error occurred on initialized ports, abort" << std::endl;
      exit(1);
    }
    struct ether_addr ether_address;
    rte_eth_macaddr_get((uint8_t) eth_dev_id, &ether_address);
    mac_addresses.push_back(ether_address);
  }
}

int DataPlane::CreateMbufPool() {
  // Create mbuf pool
  pkt_mbuf_pool = rte_pktmbuf_pool_create("dataplane_mem_pool", NUM_BYTES_MBUF,
                                          MEMPOOL_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
  if (pkt_mbuf_pool == nullptr) {
    std::cerr << "Can't initialize memory buffer pool, rte_errno : " << rte_strerror(-rte_errno) << std::endl;
    return -1;
  }
  return 0;
}

void DataPlane::InitializePortConf() {
  // Initialize port configuration options
  // In RX,
  // 1. Header Split disabled.
  // 2. IP checksum offload disabled.
  // 3. VLAN filtering disabled.
  // 4. Jumbo Frame Support disabled.
  // 5. CRC stripped by hardware.
  port_conf.rxmode.split_hdr_size = 0;
  port_conf.rxmode.header_split = 0;
  port_conf.rxmode.hw_ip_checksum = 0;
  port_conf.rxmode.hw_vlan_filter = 0;
  port_conf.rxmode.hw_vlan_strip = 0;
  port_conf.rxmode.hw_vlan_extend = 0;
  port_conf.rxmode.jumbo_frame = 0;
  port_conf.rxmode.hw_strip_crc = 0;
  // Single queue mode
  port_conf.txmode.mq_mode = ETH_MQ_TX_NONE;
}

// Initialize ports, it's responsible for setup eth config, creating RX/TX queues and start up.
int DataPlane::InitializePorts(uint8_t port) {
  // Number of RX ring descriptors, 128
  uint16_t nb_rxd = 128;
  // Number of TX ring descriptors, 512
  uint16_t nb_txd = 512;

  int ret = -1;
  // port, number of rx queue : 1, number of tx queue : 1, port configuration.
  ret = rte_eth_dev_configure(port, 1, 1, &port_conf);
  if (ret < 0) {
    std::cerr << "Could not configure port, abort." << std::endl;
    exit(1);
  }
  // Adjust number of descriptor
  ret = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
  if (ret < 0) {
    std::cerr << "Could not adjust number of descriptors, abort." << std::endl;
    exit(1);
  }
  // Setup rx queue
  ret = rte_eth_rx_queue_setup(port, 0, nb_rxd, rte_eth_dev_socket_id(port), nullptr, pkt_mbuf_pool);
  if (ret < 0) {
    std::cerr << "Could not setup RX queue" << std::endl;
    exit(1);
  }
  // Setup tx queue
  ret = rte_eth_tx_queue_setup(port, 0, nb_txd, rte_eth_dev_socket_id(port), nullptr);
  if (ret < 0) {
    std::cerr << "Could not setup TX queue" << std::endl;
    exit(1);
  }
  // Start eth dev
  ret = rte_eth_dev_start(port);
  if (ret < 0) {
    std::cerr << "Could not start eth dev, port " << port << std::endl;
    exit(1);
  }

  // Enable promiscuous mode.
  rte_eth_promiscuous_enable(port);

  return 0;
}

void DataPlane::ServeProcessingLoop(int DataPlaneCore_t) {
  DataPlaneCore *data_plane_core;
  // Match Core type
  switch (DataPlaneCore_t) {
    case SayHelloCore_t:
      data_plane_core = new SayHelloCore();
      break;
    case DumpPacketCore_t:
      data_plane_core = new DumpPacketCore(pkt_mbuf_pool);
      break;
    case BasicForwardCore_t:
      data_plane_core = new BasicForwardCore(pkt_mbuf_pool);
      break;
    case L3ForwardCPUCore_t:
      data_plane_core = new L3ForwardCPUCore(&mac_addresses);
      break;
    default:
      std::cerr << "No matching Core, abort" << std::endl;
      exit(1);
  }
  data_plane_core->LCoreFunctions();
}

} // namespace gpuflow
