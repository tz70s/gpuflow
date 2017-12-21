/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include "gpuflow/dataplane.h"
#include "gpuflow/dataplane_processor.h"

int main(int argc, char *argv[]) {
  // Create a gpu accelerated data plane
  gpuflow::DataPlane data_plane(argc, argv);
  data_plane.DisplayInfo();
  gpuflow::DataPlaneProcessor *processor = new gpuflow::SayHelloProcessor;
  data_plane.ServeProcessingLoop(processor);
  while (true) {}
  return 0;
}