#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

import os
from subprocess import call

class GPUFlowInstall:

    def __init__(self, path):
        self.path = os.path.abspath(path)

    def compile(self):
        os.chdir(self.path)
        cmake_file = self.path + '/CMakeLists.txt'
        build_folder = self.path + '/build'
        if os.path.isfile(cmake_file):
            if os.path.isdir(build_folder):
                pass
            else:
                os.mkdir('build')
            os.chdir(self.path + '/build')
            call(['cmake', '..'])
            call(['make'])

        else:
            print 'The CMakeLists.txt is not existed in this workspace, do you specify the correct path?'
            exit(1)

