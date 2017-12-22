#! /bin/bash
#
# Copyright 2017 of original authors and authors.
#
# We use MIT license for this project, checkout LICENSE file in the root of source tree.
#

# Cutting from dpdk setup script for customizing, need to add the re-distribution license.

HUGE_PAGE_SIZE = `cat /proc/meminfo  | grep Hugepagesize | cut -d : -f 2 | tr -d ' '`

# Creates hugepage filesystem.
create_mnt_huge() {
    echo "Creating /mnt/huge and mounting as hugetlbfs"
    sudo mkdir -p /mnt/huge
    grep -s '/mnt/huge' /proc/mounts > /dev/null
    if [ $? -ne 0 ] ; then
        sudo mount -t hugetlbfs nodev /mnt/huge
    fi
}

# Removes hugepage filesystem.
remove_mnt_huge() {
    echo "Unmounting /mnt/huge and removing directory"
    grep -s '/mnt/huge' /proc/mounts > /dev/null
    if [ $? -eq 0 ] ; then
        sudo umount /mnt/huge
    fi
    
    if [ -d /mnt/huge ] ; then
        sudo rm -R /mnt/huge
    fi
}

# Removes all reserved hugepages.
clear_huge_pages() {
    echo > .echo_tmp
    for d in /sys/devices/system/node/node? ; do
        echo "echo 0 > $d/hugepages/hugepages-${HUGEPGSZ}/nr_hugepages" >> .echo_tmp
    done
    echo "Removing currently reserved hugepages"
    sudo sh .echo_tmp
    rm -f .echo_tmp
    
    remove_mnt_huge
}

# Creates hugepages.
set_non_numa_pages() {
    clear_huge_pages

    echo ""
    echo "  Input the number of ${HUGEPGSZ} hugepages"
    echo "  Example: to have 128MB of hugepages available in a 2MB huge page system,"
    echo "  enter '64' to reserve 64 * 2MB pages"
    echo -n "Number of pages: "
    # TODO: Use the hard-coded pages
    read Pages

    echo "echo $Pages > /sys/kernel/mm/hugepages/hugepages-${HUGEPGSZ}/nr_hugepages" > .echo_tmp

    echo "Reserving hugepages"
    sudo sh .echo_tmp
    rm -f .echo_tmp

    create_mnt_huge
}

# Main entry
if [ $1 -eq ""] ; then
    echo "Not enough argumments!"
else
    set_non_numa_pages
fi