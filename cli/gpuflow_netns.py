#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

from gpuflow_sudo import check_sudo
from subprocess import check_output
import os
import sys
import re

class NetworkNameSpace:

    def __init__(self, target_tap):
        self.target_tap = []
        self.target_tap = target_tap.split(',')
        self.target_taps_num = len(self.target_tap)
        for i in range(0, self.target_taps_num):
            self.target_tap[i] = self.target_tap[i].split('=')
            if len(self.target_tap[i]) == 1:
                self.target_tap[i].append('')
                self.target_tap[i].append('')
            elif len(self.target_tap[i]) == 2:
                if self.target_tap[i][1] == '' or self.target_tap[i][1] == '-':
                    self.target_tap[i][1] = ''
                    self.target_tap[i].append('')
                elif self.target_tap[i][1].find('-') != -1:
                    if len(self.target_tap[i][1].split('-')) == 2:
                        self.target_tap[i].append(self.target_tap[i][1].split('-')[1])
                        self.target_tap[i][1] = self.target_tap[i][1].split('-')[0]
                    else:
                        print('ipv4 or ipv6 address format error, format of ip setting : -d <tapname>=[ipv4]-[ipv6],<tapname>=[ipv4]-[ipv6], ...')
            else:
                print('ipv4 or ipv6 address format error, format of ip setting : -d <tapname>=[ipv4]-[ipv6],<tapname>=[ipv4]-[ipv6], ...')

    def create_ns(self):
        """
        Create nents.
        """
        # Sudo permission required
        check_sudo()
        with open(os.devnull, 'wb') as devnull_:
            for i in range(0, self.target_taps_num):
                try:
                    check_output(['ip', 'netns', 'add', self.target_tap[i][0] + '-ns'], stderr=devnull_)
                    print('Create netns:' + self.target_tap[i][0] + '-ns')
                except Exception as e:
                    # unsafely ignore file existed
                    print(self.target_tap[i][0] + '-ns already exist.')
                    pass
                try:
                    check_output(['ip', 'link', 'set', self.target_tap[i][0], 'netns', self.target_tap[i][0] + '-ns'], stderr=sys.stderr)
                    print('Set ' + self.target_tap[i][0] + ' to netns ' + self.target_tap[i][0] + '-ns')
                except Exception as e:
                    raise Exception('Can not set ' + self.target_tap[i][0] + ' to ' + self.target_tap[i][0] + '-ns, does it existed?')
        
    def set_route(self, source_tap):
        """
        Set route to either target ns and self.
        """
        check_output(['ifconfig', 'dtap0', '17.0.2.3', 'up'])
        check_output(['ifconfig', 'dtap0', 'inet6', 'add', '1111:000a:0a0a:0a0a:0a0a:0a0a:0a0a:0a0a/16', 'up'])
        for i in range(0, self.target_taps_num):
            try:
                if self.target_tap[i][1] == '':
                    postfix_num = re.search('[^a-z]*[0-9]', self.target_tap[i][0])
                    self.target_tap[i][1] = '17.' + str(int(postfix_num.group())) + '.2.3'
                if self.target_tap[i][2] == '':
                    postfix_num = re.search('[^a-z]*[0-9]', self.target_tap[i][0])
                    self.target_tap[i][2] = '1111:0' + str(int(postfix_num.group())) + '0a:0a0a:0a0a:0a0a:0a0a:0a0a:0a0a'
                check_output(['ip', 'netns', 'exec', self.target_tap[i][0] + '-ns', 'ifconfig', self.target_tap[i][0], self.target_tap[i][1], 'up'])
                check_output(['ip', 'netns', 'exec', self.target_tap[i][0] + '-ns', 'ifconfig', self.target_tap[i][0], 'inet6', 'add', self.target_tap[i][2] + '/16', 'up'])
                print('Set ' + self.target_tap[i][0] + ' with ip4:' + self.target_tap[i][1] + ', ip6:' + self.target_tap[i][2] + '/16')
                check_output(['ip', 'netns', 'exec', self.target_tap[i][0] + '-ns', 'route', 'add', '-net', '0.0.0.0', 'dev', self.target_tap[i][0]])
                check_output(['ip', 'netns', 'exec', self.target_tap[i][0] + '-ns', 'route', '-A', 'inet6', 'add', '::/0', 'dev', self.target_tap[i][0]])
                check_output(['route', 'add', self.target_tap[i][1], 'dev', source_tap])
                check_output(['route', '-A', 'inet6', 'add', self.target_tap[i][2], 'dev', source_tap])
            except Exception:
                raise Exception('Can not set route, do you check neither ' + self.target_tap[i][0] + ' or ' + source_tap + ' existed?')

