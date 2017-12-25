#!/bin/bash
#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

./build/bin/GPUFlow -l 0-3 --vdev=net_tap0 --vdev=net_tap1 --vdev=net_tap2 --vdev=net_tap3 --master-lcore 0
