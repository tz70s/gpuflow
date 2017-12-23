#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

import os

# For checking sudo permission util
def check_sudo():
    if os.getuid() == 0:
        pass
    else:
        print 'Must used with sudo permission'
        exit(1)

