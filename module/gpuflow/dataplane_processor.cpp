/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include <rte_lcore.h>
#include <iostream>
#include "dataplane_processor.h"

namespace gpuflow {

int SayHelloProcessor::TypeOf() {
  return TypeSayHelloProcessor;
}

int SayHelloProcessor::LCoreFunction(__attribute__((unused)) void *args) {
  unsigned int self_lcore_id = rte_lcore_id();
  std::cout << "Hi, I'm " << self_lcore_id << " lcore" << std::endl;
  return 0;
}

} // namespace gpuflow
