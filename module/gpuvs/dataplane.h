/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _DATA_PLANE_H_
#define _DATA_PLANE_H_

namespace gpuvs {

// The data plane is responsible for initialize dpdk related attributes and functions.
// Some of things that we need to care about: e.g. num of (logic)cores, network devices, etc.
// After initialize phase, we'll have a startup functions which polymorphic calls up packet processing implementation.
// e.g. data_plane.start(l2fwd);
class DataPlane final {
 public:
  DataPlane(int argc, char *argv[], unsigned int num_of_cores = 4);

  // Display the information, configuration of core.
  void DisplayInfo();

 private:
  unsigned int num_of_cores;

  // Allocate kernel network interfaces for using.
  // We'll create two nics for ingress and egress, also, adapt to traffic generator for benchmarking.
  void AllocKernelNIC();
};

} // namespace gpuvs

#endif // _DATA_PLANE_H_
