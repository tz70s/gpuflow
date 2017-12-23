#!/bin/bash
#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

function add_dtap() {
	test -e /proc/net/pktgen/dtap1
	if [ $? -ne 0 ] ; then
 	       echo "add_device dtap1" > /proc/net/pktgen/kpktgend_0
	fi
}

function set_dtap() {
	echo "min_pkt_size 64" > /proc/net/pktgen/dtap1
	echo "max_pkt_size 9000" > /proc/net/pktgen/dtap1
	echo "count 20" > /proc/net/pktgen/dtap1
}

function start_dtap() {
	echo "start" > /proc/net/pktgen/pgctrl
}

add_dtap
set_dtap
start_dtap
