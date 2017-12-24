#!/bin/bash
#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

TARGET_IP=$1
function add_dtap() {
    test -e /proc/net/pktgen/dtap0
    if [ $? -ne 0 ] ; then
 	    echo "add_device dtap0" > /proc/net/pktgen/kpktgend_0
    fi
}

function set_dtap() {
    echo "min_pkt_size 64" > /proc/net/pktgen/dtap0
    echo "max_pkt_size 9000" > /proc/net/pktgen/dtap0
    echo "count 20" > /proc/net/pktgen/dtap0
    echo "dst ${TARGET_IP}" > /proc/net/pktgen/dtap0
}

function start_dtap() {
    echo "start" > /proc/net/pktgen/pgctrl
}

if [[ $# -lt 1 ]]; then
    echo "Invalid argument, should specify the ip address"
	exit 1
fi

add_dtap
set_dtap
start_dtap
