#!/bin/bash
#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

SOURCE_TAP=$1
TARGET_TAP=$2
TARGET_NS=${TARGET_TAP}-ns
TARGET_IP=$3

function add_netns() {
    {
        # silent
        ip netns add $TARGET_NS
    }&>/dev/null
    ip link set $TARGET_TAP netns $TARGET_NS 
}

function set_route() {
    ip netns exec $TARGET_NS ifconfig $TARGET_TAP $TARGET_IP up
    ip netns exec $TARGET_NS route add -net 0.0.0.0 dev $TARGET_TAP
    route add $TARGET_IP dev $SOURCE_TAP
}

if [[ $# -ne 3 ]]; then
    echo "Illegal number of parameters."
    echo "sudo ./set_netns <source_tap> <target_tap> <target_ip>"
    exit 1
fi

add_netns

if [ $? -eq -1 ]; then 
    echo "Ensure the sudo permission and tap ifaces existed."
    exit 1
fi

set_route
