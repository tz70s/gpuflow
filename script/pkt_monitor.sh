#!/bin/bash

INTERVAL="1"

function monitor() {
    RPKT1=`cat /sys/class/net/$TAP/statistics/rx_packets`
    RDROPPED1=`cat /sys/class/net/$TAP/statistics/rx_dropped`
    TPKT1=`cat /sys/class/net/$TAP/statistics/tx_packets`
    TDROPPED1=`cat /sys/class/net/$TAP/statistics/tx_dropped`
    sleep $INTERVAL
    RPKT2=`cat /sys/class/net/$TAP/statistics/rx_packets`
    RDROPPED2=`cat /sys/class/net/$TAP/statistics/rx_dropped`
    TPKT2=`cat /sys/class/net/$TAP/statistics/tx_packets`
    TDROPPED2=`cat /sys/class/net/$TAP/statistics/tx_dropped`
    RXDiff=`expr $RPKT2 + $RDROPPED2 - $RPKT1 - $RDROPPED1`
    TXDiff=`expr $TPKT2 + $TDROPPED2 - $TPKT1 - $TDROPPED1`
    echo "$1 TX: $TXDiff packets/sec RX: $RXDiff packets/sec"
}

if [[ $# -ne 1 ]]; then
    echo "Illegal number of parameters."
    echo "sudo ./pkt_monitor.sh <monitored_tap>"
    exit 1
fi

test -e /sys/class/net/$1/statistics/rx_bytes
if [ $? -ne 0 ]; then
    echo "$1 does not exist."
    exit 1
else
    TAP=$1
fi

while true
do
    monitor
done
