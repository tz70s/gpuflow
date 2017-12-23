#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

from gpuflow_sudo import check_sudo
from subprocess import check_output
import os
import sys

class NetworkNameSpace:

    def __init__(self, target_tap, target_ip):
        self.target_tap = target_tap
        self.target_ip = target_ip
        self.target_ns = self.target_tap + '-ns'

    def create_ns(self):
        """
        Create nents.
        """
        # Sudo permission required
        check_sudo()
        with open(os.devnull, 'wb') as devnull_:
            try:
                check_output(['ip', 'netns', 'add', self.target_ns], stderr=devnull_)
            except Exception as e:
                # unsafely ignore file existed
                pass
            try:
                check_output(['ip', 'link', 'set', self.target_tap, 'netns', self.target_ns], stderr=sys.stderr)
            except Exception as e:
                raise Exception('Can not set ' + self.target_tap + ' to ' + self.target_ns + ', does it existed?')
        
    def set_route(self, source_tap):
        """
        Set route to either target ns and self.
        """
        try:
            check_output(['ip', 'netns', 'exec', self.target_ns, 'ifconfig', self.target_tap, self.target_ip, 'up'])
            check_output(['ip' ,'netns', 'exec', self.target_ns, 'route', 'add', '-net', '0.0.0.0', 'dev', self.target_tap])
            check_output(['route' ,'add', self.target_ip, 'dev', source_tap])
        except Exception:
            raise Exception('Can not set route, do you check neither ' + self.target_tap + ' or ' + source_tap + ' existed?')

