/*
 * Copyright 2017 of original authors and authors.
 *
 * We use MIT license for this project, checkout LICENSE file in the root of source tree.
 */

#include "gpuvs/dataplane.h"

int main(int argc, char *argv[]) {

  gpuvs::DataPlane data_plane(argc, argv);
  data_plane.DisplayInfo();

  return 0;
}