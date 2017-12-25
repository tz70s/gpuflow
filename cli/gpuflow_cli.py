#!/usr/bin/python
#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

import click
import subprocess
from gpuflow_cgroup import Cgroup
from gpuflow_netns import NetworkNameSpace
from gpuflow_run import GPUFlowRun
from gpuflow_install import GPUFlowInstall

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
def run(workspace):
    gflow = GPUFlowRun(workspace)
    gflow.run()

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
@click.option('--target_tap', '-d', type=click.STRING, help='target tap device', required=True)
@click.option('--target_ip', '-i', default='17.1.2.3', type=click.STRING, help='target ip address in network namespace')
def netns(source_tap, target_tap, target_ip):
    ns = NetworkNameSpace(target_tap, target_ip)
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
    click.echo('Set up netns ' + target_tap + '-ns at ' + target_ip)

if __name__ == '__main__':
    MainGroup()

