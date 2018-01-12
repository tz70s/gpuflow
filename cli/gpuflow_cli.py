#!/usr/bin/python
#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

import click
import subprocess
import threading
import os
import time
from gpuflow_cgroup import Cgroup
from gpuflow_netns import NetworkNameSpace
from gpuflow_run import GPUFlowRun
from gpuflow_install import GPUFlowInstall
from gpuflow_hugepage import Hugepage
from gpuflow_pktmont import PacketMonitor
from gpuflow_pktgen import PacketGenerator

@click.group()
def MainGroup():
    pass

@MainGroup.command(help='install(compile), pass the root of source directory as workspace')
@click.argument('workspace')
def install(workspace):
    gflow = GPUFlowInstall(workspace)
    gflow.compile()

@MainGroup.command(help='run the GPUFlow program, pass the root of source directory as workspace')
@click.argument('workspace')
@click.option('--limit', '-l', default='0', help='limitation of cpu consumption, must enclosed in 0-1')
@click.option('--monitor', '-m', type=click.STRING, default='not', help='Monitor dtap multi-dtap use \',\' to split ex: dtap0,dtap1, set \'all\' to monitor all dtap at once.')
def run(workspace, limit, monitor):
    gflow = GPUFlowRun(workspace)

    record = []
    lock  = threading.Lock()
    mainup = [-1]

    def mainthread(pid):
        mainup[0] = os.getpid()
        gflow.run()
        mainup[0] = -1

    def autocgroup():
        while (mainup[0] == -1) :
            continue
        cg = Cgroup(float(limit))
        try:
            cg.create_cg()
        except Exception as e:
            print e.message
            exit(1)
        click.echo('Set up cgroup limitation : ' + limit)

    def automonitor():
        while (mainup[0] == -1) or not os.path.exists('/sys/class/net/dtap0'):
            continue
        pm = PacketMonitor(monitor, 1)
        try:
            pm.monitor(mainup)
        except Exception as e:
            print e.message
            exit(1)

    thread = threading.Thread(target=mainthread,args=(mainup))
    thread.start()
    record.append(thread)

    if float(limit) != 0:
        thread1 = threading.Thread(target=autocgroup,args=())
        thread1.start()
        record.append(thread1)

    if monitor != 'not':
        thread2 = threading.Thread(target=automonitor,args=())
        thread2.start()
        record.append(thread2)

    for thread in record:
        thread.join()


@MainGroup.command(help = 'Set up cgroup in a limitation of cpu consumption')
@click.option('--limit', '-l', default='0.5', help='limitation of cpu consumption, must enclosed in 0-1')
def cgroup(limit):
    cg = Cgroup(float(limit))
    try:
        cg.create_cg()
    except Exception as e:
        print e.message
        exit(1)
    click.echo('Set up cgroup limitation : ' + limit)

@MainGroup.command(help = 'Set up netns for forwarding application test')
@click.option('--source_tap', '-s', type=click.STRING, help='source tap device', required=True)
@click.option('--target_tap', '-d', type=click.STRING, help='target tap devices, format of ip setting : -d <tapname>=[ipv4]-[ipv6],<tapname>=[ipv4]-[ipv6], ...', required=True)
def netns(source_tap, target_tap):
    ns = NetworkNameSpace(target_tap)
    try:
        ns.create_ns()
    except Exception as e:
        print e.message
        exit(1)
    try:
        ns.set_route(source_tap)
    except Exception as e:
        print e.message
        exit(1)
    click.echo('Set up netnss')

@MainGroup.command(help = 'Packet generator of tap')
@click.option('--generator_tap', '-g', default='dtap0', type=click.STRING, help='The tap to generate packets, multi-tap use \',\' to split eg: dtap0,dtap1.')
@click.option('--min_pkt_size', '-mi', default='64', type=click.STRING, help='Minimum size of packet.')
@click.option('--max_pkt_size', '-ma', default='64', type=click.STRING, help='Maximum size of packet.')
@click.option('--count', '-c', default='0', type=click.STRING, help='Sets number of packets to send, set to zero for continuous sends until explicitly stopped.')
@click.option('--ipv', '-v', default='ipv6', type=click.STRING, help='Version of ip insert \'ipv6\' or \'ipv4\'.')
@click.option('--dst_ip', '-d', default='1111:010a:0a0a:0a0a:0000:0000:0000:0000~1111:010a:0a0a:0a0a:ffff:ffff:ffff:ffff', type=click.STRING, help='Random the generated ip address from <start ip> to <end ip> format by -d <start ip>~<end ip> or use single ip -d <dst ip>.')
@click.option('--ratep', '-r', default='2500000', type=click.STRING, help='set rate to <integer>pps')
def pktgen(generator_tap, min_pkt_size, max_pkt_size, count, ipv, dst_ip, ratep):
    pg = PacketGenerator(generator_tap, min_pkt_size, max_pkt_size, count, ipv, dst_ip, ratep)
    try:
        pg.generate()
    except Exception as e:
        print e.message
        exit(1)
    click.echo('Generating packets with ' + generator_tap)

@MainGroup.command(help = 'Monitor the dtaps\' TX and RX')
@click.option('--monitor_dtap', '-m', default='dtap0,dtap1,dtap2,dtap3', type=click.STRING, help='Monitor dtap, multi-dtap use \',\' to split ex: dtap0,dtap1, set \'all\' to monitor all dtap at once.', required=True)
@click.option('--interval', '-i', default='1', type=click.INT, help='The monitor interval')
def pktmont(monitor_dtap, interval):
    pm = PacketMonitor(monitor_dtap, int(interval))
    try:
        pm.monitor([1])
    except Exception as e:
        print e.message
        exit(1)
    click.echo('Monitoring ' + monitor_dtap + ' packets')

def removehu(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    number = 0
    try:
        hu = Hugepage(number)
    except Exception as e:
        print e.message
        exit(1)
    click.echo('clean the hugepage mapping.')
    ctx.exit()

@MainGroup.command(help = 'Set up non-NUMA hugepage for DPDK.')
@click.option('--remove', '-r', is_flag=True, callback=removehu, expose_value=False, is_eager=True, help='remove hugepage')
@click.option('--number', '-n', default='1024', type=click.INT, help='number of ' + Hugepage.size() + ' hugepages, default 1024')
def hugepage(number):
    hu = Hugepage(int(number))
    try:
        hu.set_non_numa_pages()
    except Exception as e:
        print e.message
        exit(1)
    click.echo('Set up ' + str(number) + ' ' + Hugepage.size() + ' non-NUMA hugepage')

if __name__ == '__main__':
    MainGroup()

