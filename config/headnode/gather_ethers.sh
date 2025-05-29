#!/bin/bash

# Script to ping all of the ndp*-data* hosts in the `hosts` file
# and find the MAC address associated with each of them.
# 
# Usage: ./gather_ethers.sh <hostfile>

filename="../hosts"
if [[ "$1" != "" ]]; then
        filename=$1
fi

ips=$(grep ndp ${filename} | grep data | awk '{print $1}')
for ip in ${ips}; do
        ping -c1 ${ip} > /dev/null 2> /dev/null
done

macs=$(arp -v | grep ndp | grep data | sort)
old_IFS=${IFS}
IFS=$'\n'
for mac in ${macs}; do
        echo ${mac} | awk '{print $3,$1}'
done
IFS=${old_IFS}

