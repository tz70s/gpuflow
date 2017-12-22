# Gpu Accelerated Flow Re-Routing

## Installation

### Pre-requisition

```bash
sudo apt-get update
sudo apt-get install -y make coreutils gcc gcc-multilib libnuma-dev python
```

### Important configuration

Dpdk depends on huge page and vfio/uio, etc. You need to prepare these settings up.
Checkout `script/setup.sh` for huge page settings. (However, it's not usable, currently)

We can check out `usertools/dpdk-setup.sh` for open options.
It's more convenient, even help us to build up sources.
i.e. 
1. choose the `x86_64-native-linuxapp-gcc` option
2. open hugepage options, recommend value -> 1024

### Compile from source

```bash
wget http://fast.dpdk.org/rel/dpdk-17.05.2.tar.xz
tar xf dpdk-17.05.2.tar.xz
# Set up necessary environment variable
export RTE_SDK=$PWD/dpdk-stable-17.05.2
export RTE_TARGET=x86_64-native-linuxapp-gcc

# Go to the dpdk source folder
cd $RTE_SDK

# Set up config
# After doing this, the config is loacted at x86_64-native-linuxapp-gcc/.config
make config T=x86_64-native-linuxapp-gcc
make

# Go to the source folder
cd /path/to/this/source/folder

# Currently, we don't specify the build path of dpdk as environment variable in cmake.
# Make sure CMakeLists.txt in the root of repo specify the correct path to dpdk.
./configure

# Execution
sudo ./run

# Or
sudo ./build/GPUFlow --vdev=net_tap0 --vdev=net_tap1 --vdev=net_tap2 --vdev=net_tap3

# Open another terminal
ifconfig -a
```

### LICENSE 

MIT license, the program is redistributed from dpdk with BSD license.
