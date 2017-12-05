# Gpu Accelerated Virtual Switch

## Installation

### Pre-requisition

```bash
sudo apt-get update
sudo apt-get install -y make coreutils gcc gcc-multilib libnuma-dev python
```

### Important configuration

Dpdk depends on huge page and vfio/uio, etc. You need to prepare these settings up.
Checkout `script/setup.sh` for huge page settings.

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

# Compile
# Specify a local build place
export RTE_TARGET=build
make
```

### LICENSE 

MIT license, the program is redistributed from dpdk with BSD license.