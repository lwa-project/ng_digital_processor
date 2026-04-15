#!/bin/bash

# Script to ping all of the ndp*-data* hosts in the `hosts` file
# and find the MAC address associated with each of them.
# 
# Usage: ./gather_ethers.sh <hostfile>

filename="../hosts"
if [[ "$1" != "" ]]; then
  filename=$1
fi

if [[ ! -f ${filename} ]]; then
	echo "Host file '${filename}' does not exist"
	exit
fi

ips=$(grep ndp0-data ${filename} | awk '{print $1}')
macs0=""
for ip in ${ips}; do
  hname=$(grep ${ip} ${filename} | awk '{print $2}')
  iface=$(ip addr show to ${ip} | head -n1 | awk '{print $2}' | sed -e "s/:.*//g;")
  mac=$(ip link show ${iface} | grep ether | awk '{print $2}')
  echo "${mac} ${hname}"
done

ips=$(grep ndp ${filename} | grep data | awk '{print $1}')
for ip in ${ips}; do
  ping -c1 ${ip} > /dev/null 2> /dev/null
done

macs=$(arp -v | grep ndp | grep data | awk '{print $1,$3}' | sort | uniq)
old_IFS=${IFS}
IFS=$'\n'
for mac in ${macs}; do
  echo ${mac} | awk '{print $2,$1}'
done
IFS=${old_IFS}
