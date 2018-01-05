/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include <iostream>
#include <rte_ip.h>
#include "dataplane_lpm_gpu.h"

namespace gpuflow {

DataPlaneLPMIPv4GPU::DataPlaneLPMIPv4GPU(){
  if (CreateLPMTable() < 0) {
    std::cerr << "Error occured on creating lpm table" << std::endl;
    exit(1);
  }
}

int DataPlaneLPMIPv4GPU::CreateLPMTable() {
  if (ipv4_lpm_factory.CreateLPMTable() < 0) {
    std::cerr << "Create lpm table error from ipv4 lpm factory" << std::endl;
    exit(1);
  }
  ipv4_lpm_factory.AddLPMRule(IPv4(17, 0, 0, 0), 16, 0);
  ipv4_lpm_factory.AddLPMRule(IPv4(17, 1, 0, 0), 16, 1);
  ipv4_lpm_factory.AddLPMRule(IPv4(17, 2, 0, 0), 16, 2);
  ipv4_lpm_factory.AddLPMRule(IPv4(17, 3, 0, 0), 16, 3);
  if (ipv4_lpm_factory.IPv4TBL24 != nullptr) {
    IPv4TBL24 = ipv4_lpm_factory.IPv4TBL24;
  } else {
    std::cerr << "The IPv4TBL24 pointer isn't existed" << std::endl;
    exit(1);
  }
  return 0;
}

DataPlaneLPMIPv6GPU::DataPlaneLPMIPv6GPU() {
  if(CreateLPMTable() < 0) {
    std::cerr << "Error occured on creating ipv6 lpm table" << std::endl;
    exit(1);
  }  
}

int DataPlaneLPMIPv6GPU::CreateLPMTable() {
  if(ipv6_lpm_factory.CreateLPMTable() < 0) {
    std::cerr << "Create lpm table error from ipv6 lpm factory";  
    exit(1);
  } else {
    std::cout << "Create lpm table from ipv6 lpm factory successfully." << std::endl;
  }
  if (ipv6_lpm_factory.IPv6TBL24 != nullptr) {
    IPv6TBL24 = ipv6_lpm_factory.IPv6TBL24;
    std::cout << "The IPv6TBL24 pointer is existed" << std::endl;
  } else {
    std::cerr << "The IPv6TBL24 pointer isn't existed" << std::endl;
    exit(1);
  }
  return 0;
}

} // namespace gpuflow
