#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#
import subprocess
import commands
import os
import time
import multiprocessing
from gpuflow_sudo import check_sudo

class PacketMonitor:

    def __init__(self, monitor_dtap, interval):
        """
        Clear the old setting first
        """
        self.monitor_dtap = monitor_dtap
        self.interval = int(interval)

    def monitor(self, mainpid):
        """
        Montior with check the statistics files
        """
        # check sudo permission
        check_sudo()
        
        if self.monitor_dtap != 'all':
            dtaps = []
            dtaps = self.monitor_dtap.split(',')
            self.getstate(dtaps, mainpid)
        else:
            dtaps = []
            for dtap_exist in range(0, int(multiprocessing.cpu_count())):
                if os.path.exists('/sys/class/net/dtap' + str(dtap_exist)):
                    dtaps.append('dtap' + str(dtap_exist))
                    print('device dtap' + str(dtap_exist) + ' found.')
            self.getstate(dtaps, mainpid)

    def getstate(self, dtaps, mainpid):
        """
        Get and calculate the state data
        """
        dtap_num = len(dtaps)
        str_tx_pkt = [ 0 for i in range(dtap_num) ]
        str_tx_drp = [ 0 for i in range(dtap_num) ]
        str_rx_pkt = [ 0 for i in range(dtap_num) ]
        str_rx_drp = [ 0 for i in range(dtap_num) ]
        end_tx_pkt = [ 0 for i in range(dtap_num) ]
        end_tx_drp = [ 0 for i in range(dtap_num) ]
        end_rx_pkt = [ 0 for i in range(dtap_num) ]
        end_rx_drp = [ 0 for i in range(dtap_num) ]
        # Start monitor
        while mainpid[0] != -1:
            i = 0
            for dtap in dtaps:
                # Test wheater the dtap in namespace
                result, ns_tap_exist = commands.getstatusoutput('sudo ip netns exec ' + dtap + '-ns ifconfig | grep ' + dtap)
                if os.path.exists('/sys/class/net/' + dtap):
                    ns_prefix = ''
                elif ns_tap_exist.strip(' ') != '':
                    ns_prefix = 'ip netns exec ' + dtap + '-ns '
                else:
                    print(dtap + ' not found')
                    next
                result, str_tx_pkt[i] = commands.getstatusoutput(ns_prefix + 'cat /sys/class/net/' + dtap + '/statistics/tx_packets')
                result, str_tx_drp[i] = commands.getstatusoutput(ns_prefix + 'cat /sys/class/net/' + dtap + '/statistics/tx_dropped')
                result, str_rx_pkt[i] = commands.getstatusoutput(ns_prefix + 'cat /sys/class/net/' + dtap + '/statistics/rx_packets')
                result, str_rx_drp[i] = commands.getstatusoutput(ns_prefix + 'cat /sys/class/net/' + dtap + '/statistics/rx_dropped')
                i += 1
            time.sleep(self.interval)
            i = 0
            for dtap in dtaps:
                # Test wheater the dtap in namespace
                result, ns_tap_exist = commands.getstatusoutput('sudo ip netns exec ' + dtap + '-ns ifconfig | grep ' + dtap)
                if os.path.exists('/sys/class/net/' + dtap):
                    ns_prefix = ''
                elif ns_tap_exist.strip(' ') != '':
                    ns_prefix = 'ip netns exec ' + dtap + '-ns '
                else:
                    print(dtap + ' not found')
                    continue
                result, end_tx_pkt[i] = commands.getstatusoutput(ns_prefix + 'cat /sys/class/net/' + dtap + '/statistics/tx_packets')
                result, end_tx_drp[i] = commands.getstatusoutput(ns_prefix + 'cat /sys/class/net/' + dtap + '/statistics/tx_dropped')
                result, end_rx_pkt[i] = commands.getstatusoutput(ns_prefix + 'cat /sys/class/net/' + dtap + '/statistics/rx_packets')
                result, end_rx_drp[i] = commands.getstatusoutput(ns_prefix + 'cat /sys/class/net/' + dtap + '/statistics/rx_dropped')
                tx_diff = (int(end_tx_pkt[i]) + int(end_tx_drp[i])) - (int(str_tx_pkt[i]) + int(str_tx_drp[i]))
                rx_diff = (int(end_rx_pkt[i]) + int(end_rx_drp[i])) - (int(str_rx_pkt[i]) + int(str_rx_drp[i]))
                print(dtap + ' TX: ' + str(tx_diff) + ' packets/sec RX: ' + str(rx_diff) + ' packets/sec')
                i += 1
            print('-------------------------------------------------------')
