#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

import subprocess
import commands
import os
from gpuflow_sudo import check_sudo

class Hugepage:

    def __init__(self, number):
        """
        Clear the old setting first
        """
        self.hppath_prefix = '/sys/devices/system/node/node'
        self.number = number
        self.clear_huge_pages()

    @staticmethod
    def size():
        return str(subprocess.check_output('cat /proc/meminfo  | grep Hugepagesize | cut -d : -f 2 | tr -d \' \'', shell=True)).split('\n')[0]

    def clear_huge_pages(self):
        """
        Clear hugepage mapping before setting
        """
        # check sudo permission
        check_sudo()
        nodei = 0
        while True:
            # setting hugepage file dir
            hp_dir = self.hppath_prefix + str(nodei) + '/hugepages/hugepages-' + self.size()
            # check if node# exist then setting hugepage
            if os.path.exists(hp_dir):
                # cleanup the hugepage number
                with open(hp_dir + '/nr_hugepages', 'w') as hp_number:
                    hp_number.write('0')
                nodei += 1
            else:
                break
        print 'Removing currently reserved hugepages'
        self.remove_mnt_huge()
    
    def remove_mnt_huge(self):
        """
        Removes hugepage filesystem.
        """
        print 'Unmounting /mnt/huge and removing directory'
        result, grep_output = commands.getstatusoutput('grep \'/mnt/huge\' /proc/mounts')
        if grep_output != '':
            result, cmd_out = commands.getstatusoutput('umount /mnt/huge')
        if os.path.exists('/mnt/huge'):
            result, cmd_out = commands.getstatusoutput('rm -R /mnt/huge')

    def set_non_numa_pages(self):
        """
        Creates hugepage .
        """
        print 'Reserving ' + str(self.number) + ' of ' + self.size() + ' hugepages'
        # overwrite the hugepage number
        with open('/sys/kernel/mm/hugepages/hugepages-' + self.size() + '/nr_hugepages', 'w') as hp_number:
            hp_number.write(str(self.number))
        print 'Creating /mnt/huge and mounting as hugetlbfs'
        result, cmd_out = commands.getstatusoutput('mkdir -p /mnt/huge')
        result, grep_output = commands.getstatusoutput('grep \'/mnt/huge\' /proc/mounts')
        if grep_output == '':
            result, cmd_out = commands.getstatusoutput('mount -t hugetlbfs nodev /mnt/huge')
        

