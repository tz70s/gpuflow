#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#
from gpuflow_sudo import check_sudo
import os
import sys
import multiprocessing
import commands
import threading
class PacketGenerator:
    def __init__(self, generator_tap, min_pkt_size, max_pkt_size, count, ipv, dst_ip, ratep):
        self.generator_tap = []
        self.generator_tap = generator_tap.split(',')
        while '' in self.generator_tap: self.generator_tap.remove('')
        self.min_pkt_size = min_pkt_size
        self.max_pkt_size = max_pkt_size
        self.count = count
        if len(dst_ip.split('~')) == 2:
            self.start_ip, self.end_ip = dst_ip.split('~')
        elif len(dst_ip.split('~')) == 1:
            self.start_ip = self.end_ip = dst_ip
        else:
            print('Initalize error, ip interval is <start addr>~<end addr>')
            exit(1)
        self.ratep = ratep
        self.ipv = ipv


    def generate(self):
        """
        Setup and generate.
        """
        # Sudo permission required
        check_sudo()
        result, sys.stderr = commands.getstatusoutput('modprobe pktgen')
        # Add dtap
        cores = int(multiprocessing.cpu_count()) - 1
        for tap in self.generator_tap:
            if not os.path.exists('/proc/net/pktgen/' + tap):
                result, sys.stderr = commands.getstatusoutput('echo "add_device ' + tap + '" > /proc/net/pktgen/kpktgend_' + str(cores))
                cores -= 1
                print(sys.stderr)

        shellcode = 'while true; do '
        for tap in self.generator_tap:
            # Set dtap
            shellcode += 'echo "min_pkt_size ' + self.min_pkt_size + '" > /proc/net/pktgen/' + tap + ' | '
            shellcode += 'echo "max_pkt_size ' + self.max_pkt_size + '" > /proc/net/pktgen/' + tap + ' | '
            # count 0 means transmitting the packets continuously.
            shellcode += 'echo "count ' + self.count + '" > /proc/net/pktgen/' + tap + ' | '
            shellcode += 'echo "ratep ' + self.ratep + '" > /proc/net/pktgen/' + tap + ' | '
        
            if self.ipv == 'ipv6':
                shellcode += 'echo "dst6_min ' + self.start_ip + '" > /proc/net/pktgen/' + tap + ' | '
                shellcode += 'echo "dst6_max ' + self.end_ip + '" > /proc/net/pktgen/' + tap + ' | '
            elif self.ipv == 'ipv4':
                shellcode += 'echo "dst_min ' + self.start_ip + '" > /proc/net/pktgen/' + tap + ' | '
                shellcode += 'echo "dst_max ' + self.end_ip + '" > /proc/net/pktgen/' + tap + ' | '
            else:
                print('Wrong ip version, please insert \'ipv6\' or \'ipv4\'.')

        # Start dtap
        shellcode += 'echo "start" > /proc/net/pktgen/pgctrl'
        shellcode += '; done'
        result, sys.stderr = commands.getstatusoutput(shellcode)
