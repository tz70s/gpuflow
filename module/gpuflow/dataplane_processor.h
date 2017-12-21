/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#ifndef _DATAPLANE_PROCESSOR_H_
#define _DATAPLANE_PROCESSOR_H_

#include <rte_launch.h>

namespace gpuflow {

class DataPlaneProcessor {};

class SayHelloProcessor : public DataPlaneProcessor {
 public:
  static int LCoreFunction(__attribute__((unused)) void *args);
};

} // namespace gpuflow

#endif // _DATAPLANE_PROCESSOR_H_
