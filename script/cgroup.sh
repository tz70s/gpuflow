#!/bin/bash
#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

function create_cg() {
    mkdir -p /sys/fs/cgroup/cpu/gpuflow/
    mkdir -p /sys/fs/cgroup/cpuset/gpuflow/
}

function set_quota() {
    echo 100000 > /sys/fs/cgroup/cpu/gpuflow/cpu.cfs_period_us
    echo 70000 > /sys/fs/cgroup/cpu/gpuflow/cpu.cfs_quota_us
}

function limit() {
    for i in $(pgrep -w GPUFlow); do echo $i > /sys/fs/cgroup/cpu/gpuflow/tasks; done
}

create_cg
set_quota
limit