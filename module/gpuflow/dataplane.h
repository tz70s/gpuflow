/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _DATAPLANE_H_
#define _DATAPLANE_H_

#include <string>
#include <vector>
#include "dataplane_core.h"

namespace gpuflow {

// Forward declaration
class DataPlaneCore;

// The data plane is responsible for initialize dpdk related attributes and functions.
// Some of things that we need to care about: e.g. num of (logic)cores, network devices, etc.
// After initialize phase, we'll have a startup functions which polymorphic calls up packet processing implementation.
// e.g. data_plane.start(l2fwd);
class DataPlane {
 public:
  DataPlane(int argc, char *argv[], unsigned int num_of_cores = 4);

  // Display the information, configuration of core.
  void DisplayInfo();

  std::vector<int> * retrieve_tap_fds() { return &tap_fds; };
  // Serve the processing loop, which are different processing contents.
  void ServeProcessingLoop(int);

 private:

  unsigned int num_of_cores;
  DataPlaneCore *data_plane_core_ptr;

  // Note that, the tap names have limited size, we should carefully use it.
  std::vector<std::string> tap_names { "vtap1io", "vtap2io", "vtap3io", "vtap4io" };
  std::vector<int> tap_fds;

  // Allocate tap interfaces for read/write network packets
  void AllocTapInterface();
};

} // namespace gpuflow

#endif // _DATAPLANE_H_
