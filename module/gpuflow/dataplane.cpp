/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include "dataplane.h"
#include <iostream>

#ifdef __linux__
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <netinet/in.h>
#include <linux/if.h>
#include <linux/if_tun.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#endif

namespace gpuflow {

namespace nics {

// Create a tap network interface, or use existing one with same name.
#ifdef __linux__
static int CreateTap(char *name) {
  // Open tun file descriptor
  int tun_fd = open("/dev/net/tun", O_RDWR);
  if (tun_fd < 0) {
    std::cerr << "Can't open tun fd" << std::endl;
    exit(1);
  }
  struct ifreq ifr;
  memset(&ifr, 0, sizeof(ifr));
  // TAP device without packet information
  ifr.ifr_flags = IFF_TAP | IFF_NO_PI;
  if (name && *name) {
    snprintf(ifr.ifr_name, IFNAMSIZ, "%s", name);
  }
  int ret = ioctl(tun_fd, TUNSETIFF, (void *) &ifr);
  if (ret < 0) {
    std::cerr << "Error occured in ioctl tun/tap" << std::endl;
    close(tun_fd);
    return ret;
  }
  if (name) snprintf(name, IFNAMSIZ, "%s", ifr.ifr_name);
  return tun_fd;
}
#endif

} // namespace nics

DataPlane::DataPlane(int argc, char **argv, unsigned int num_of_cores) : num_of_cores(num_of_cores),
  data_plane_core_ptr(nullptr) {
  // Allocate tap interfaces.
  AllocTapInterface();
  data_plane_core_ptr = new DataPlaneCore(argc, argv, this, num_of_cores);
}

void DataPlane::DisplayInfo() {
  std::cout << "Display the DPDK related information : \n\t"
            << "Num of cores : " << num_of_cores
            << std::endl;
}

void DataPlane::ServeProcessingLoop(int DataPlaneProcessor_t) {
  data_plane_core_ptr->ServeProcessingLoop(DataPlaneProcessor_t);
}

void DataPlane::AllocTapInterface() {
#ifdef __linux__
  int ret = 0;
  // Create taps
  std::cout << "GPUFlow : Create taps for re-routing flows..." << std::endl;
  for (auto tap_name : tap_names) {
    ret = nics::CreateTap((char *)tap_name.c_str());
    if (ret < 0) {
      std::cerr << "Problem on creating taps" << std::endl;
      exit(1);
    } else {
      // Collect tap_fd.
      tap_fds.push_back(ret);
    }
  }
#else
  std::cerr << "Not supported platform, is linux only." << std::endl;
  exit(1);
#endif
}

} // namespace gpuflow
