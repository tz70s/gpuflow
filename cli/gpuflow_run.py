#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

from subprocess import call
from gpuflow_sudo import check_sudo
import sys
import os

class GPUFlowRun:

    def __init__(self, path):
        self.path = os.path.abspath(path)

    def run(self):
        check_sudo()
        gpuflow_bin = self.path + '/build/GPUFlow'
        if os.path.isfile(gpuflow_bin):
            call([gpuflow_bin, '-l', '0-3', '--vdev=net_tap0', '--vdev=net_tap1', '--vdev=net_tap2', 
                '--vdev=net_tap3', '--master-lcore', '0'])
        else:
            print "GPUFlow program is not existed, did you already build up?"

