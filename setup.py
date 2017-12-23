#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

from setuptools import setup, find_packages

setup(
    name='GPUFlowCLI',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        gflow=cli.gpuflow_cli:MainGroup
    ''',
)
