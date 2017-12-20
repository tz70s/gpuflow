/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include "dataplane.h"
#include <iostream>
#include <vector>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_kni.h>

namespace gpuvs {

DataPlane::DataPlane(int argc, char **argv, unsigned int num_of_cores) : num_of_cores(num_of_cores) {
  int ret = rte_eal_init(argc, argv);
  if (ret < 0) {
    rte_exit(EXIT_FAILURE, "ERROR with EAL initialization\n");
  }
  argc -= ret;
  argv += ret;
}

void DataPlane::DisplayInfo(){
  std::cout << "Display the DPDK related information : \n"
            << "Num of cores : " << num_of_cores
            << std::endl;
}

void DataPlane::AllocKernelNIC(){
  struct rte_kni *kni;
  struct rte_kni_conf conf;
  std::vector<struct kni_port_params> ports;
}

} // namespace gpuvs
