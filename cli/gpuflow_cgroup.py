#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

import subprocess
import os
from gpuflow_sudo import check_sudo

class Cgroup:

    def __init__(self, limitation):
        self.limitation = limitation
        self.cg_prefix = '/sys/fs/cgroup'

    def create_cg(self):
        """
        Create a cgroup, which named gpuflow.
        In cpu and cpuset.
        """
        # check sudo permission
        check_sudo()

        # Create cg dirs
        cg_cpu_dir = self.cg_prefix + '/cpu/gpuflow'
        cg_cpuset_dir = self.cg_prefix + '/cpuset/gpuflow'
        if not os.path.exists(cg_cpu_dir):
            os.mkdir(cg_cpu_dir)
        if not os.path.exists(cg_cpuset_dir):
            os.mkdir(cg_cpuset_dir)

        # Set up quotas
        if self.limitation > 0 and self.limitation < 1:
            with open(self.cg_prefix + '/cpu/gpuflow/cpu.cfs_period_us', 'w') as cg_period:
                cg_period.write('100000')
            quota = int(round(self.limitation * 100000))
            with open(self.cg_prefix + '/cpu/gpuflow/cpu.cfs_quota_us', 'w') as cg_quota:
                cg_quota.write(str(quota))
        else:
            raise Exception('The limitation setting is not correct! Must enclosed in 0 < limitation < 1')

        tids_list = []
        try:
            tids_list = self.retrieve_tids().splitlines()
            tasks = os.open(self.cg_prefix + '/cpu/gpuflow/tasks', os.O_WRONLY)
            for tid in tids_list:
                os.write(tasks, tid)
            os.close(tasks)
        except Exception as e:
            raise Exception('Error occurred on wrapping thread ids to cgroup')
            exit(1)
        
    def retrieve_tids(self):
        tids_str = ''
        try:
            tids_str = str(subprocess.check_output(['pgrep', '-w', 'GPUFlow']))
        except Exception as e:
            # Rethrow more clear message
            raise Exception('The GPUFlow is not started.')
        return tids_str

