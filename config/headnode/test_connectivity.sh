#!/bin/bash

# Script to make sure that all ndp hosts can be SSH'd into and test 
# various other things that will make deployment easier.
# 
# Usage: ./test_connectivity.sh <hostfile> <configfile>

filename="../hosts"
configname="../ndp_config.json"
if [[ "$1" != "" ]]; then
        filename=$1
fi
if [[ "$2" != "" ]]; then
        configname=$2
fi

if [[ ! -f ${filename} ]]; then
	echo "Host file '${filename}' does not exist"
	exit
fi
if [[ ! -f ${configname} ]]; then
	echo "NDP configuration file '${configname}' does not exist"
	exit
fi


# Part 1 - The servers

snames=$(grep ndp ${filename} | grep -v data | grep -v ipmi | awk '{print $2}' | sort)

echo "Testing password-less SSH access"
nfailed=0
for sname in ${snames}; do
        timeout 10 ssh ndp@${sname} date > /dev/null
        if [[ $? != 0 ]]; then
                nfailed=$((nfailed + 1))
                echo "  Failed on ${sname}"
        fi
done
if [[ ${nfailed} == 0 ]]; then
        echo "  OK"
fi

inames=$(grep data_ifaces ${configname} | sed -e "s/.*\[//g;s/\].*//g;s/\",*//g;")

echo "Testing interface naming"
nfailed=0
for sname in ${snames}; do
        ifaces=$(timeout 10 ssh ndp@${sname} "ip link | awk '{print \$2}'")
        declare -a tested_inames
        i=0
        for iname in ${inames}; do
                if [[ ! "${tested_inames[*]}" =~ "${iname}" ]]; then
                        found=$(echo ${ifaces} | grep ${iname})
                        if [[ "${found}" == "" ]]; then
                                nfailed=$((nfailed + 1))
                                echo "  Failed on ${sname} with iface ${iname}"
                        fi
                        tested_inames[$i]=${iname};
                        i=$((i + 1))
                fi
        done
        unset tested_inames
done
if [[ ${nfailed} == 0 ]]; then
        echo "  OK"
fi

echo "Testing headnode MACs"
nfailed=0
for sname in ${snames}; do
        if [[ "${sname:0:4}" != "ndp0" ]]; then
                macs=$(timeout 10 ssh ndp@${sname} arp -v)
                if [[ $? != 0 ]]; then
                        nfailed=$((nfailed + 1))
                        echo "  Failed to poll on ${sname}"
                else
                        iface1=$(echo ${macs} | grep ndp0-data1 | awk '{print $5}')
                        iface2=$(echo ${macs} | grep ndp0-data2 | awk '{print $5}')
                        if [[ "${iface1}" == "${iface2}" ]]; then
                                nfailed=$((nfailed + 1))
                                echo "  Failed ${sname}"
                        fi
                fi
        fi
done
if [[ ${nfailed} == 0 ]]; then
        echo "  OK"
fi

echo "Checking for 'sshpass'"
which sshpass > /dev/null
if [[ $? != 0 ]]; then
        echo "  'sshpass' not found, exiting"
        exit
else
        echo "  OK"
fi

passwd=$(grep -A5 -e 'server": {' ${configname} | grep password | sed -e "s/\",*//g;" | awk '{print $2}')

echo "Testing root SSH access"
nfailed=0
for sname in ${snames}; do
        timeout 10 sshpass -p ${passwd} ssh -o StrictHostKeyChecking=no root@${sname} date >/dev/null
        if [[ $? != 0 ]]; then
                nfailed=$((nfailed + 1))
                echo "  Failed on ${sname}"
        fi
done
if [[ ${nfailed} == 0 ]]; then
        echo "  OK"
fi

echo "Testing shared /home directory"
nfailed=0
for sname in ${snames}; do
        tname=$(mktemp -p /home/ndp)
        hdir=$(timeout 10 ssh ndp@${sname} "ls -l ${tname}")
        if [[ $? != 0 ]]; then
                nfailed=$((nfailed + 1))
                echo "  Failed to poll on ${sname}"
        else
                found=$(echo ${hdir} | grep ${tname})
                if [[ "${found}" == "" ]]; then
                        nfailed=$((nfailed + 1))
                        echo "  Failed on ${sname}"
                fi
        fi
        rm -f ${tname}
done
if [[ ${nfailed} == 0 ]]; then
        echo "  OK"
fi

echo "Testing internet access"
nfailed=0
for sname in ${snames}; do
        timeout 10 ssh ndp@${sname} "ping -c1 8.8.8.8" > /dev/null 2> /dev/null
        if [[ $? != 0 ]]; then
                nfailed=$((nfailed + 1))
                echo "  Failed on ${sname} with ping"
        else
                dns=$(timeout 10 ssh ndp@${sname} "dig fornax.phys.unm.edu")
                if [[ $? != 0 ]]; then
                        nfailed=$((nfailed + 1))
                        echo "  Failed to poll on ${sname}"
                else
                        found=$(echo ${dns} | grep -e "ANSWER SECTION")
                        if [[ "${found}" == "" ]]; then
                                nfailed=$((nfailed + 1))
                                echo "  Failed on ${sname} with dns"
                        fi
                fi
        fi
done
if [[ ${nfailed} == 0 ]]; then
        echo "  OK"
fi
        
# Part 2 - The IPMI interfaces

snames=$(grep ndp ${filename} | grep ipmi | awk '{print $2}' | sort)

usernm=$(grep -A5 -e 'ipmi": {' ${configname} | grep username | sed -e "s/\",*//g;" | awk '{print $2}')
passwd=$(grep -A5 -e 'ipmi": {' ${configname} | grep password | sed -e "s/\",*//g;" | awk '{print $2}')

echo "Checking for 'ipmitool'"
which ipmitool > /dev/null
if [[ $? != 0 ]]; then
        echo "  'ipmitool' not found, exiting"
        exit
else
        echo "  OK"
fi

echo "Testing IPMI access"
nfailed=0
for sname in ${snames}; do
        timeout 10 ipmitool -H ${sname} -U ${usernm} -P ${passwd} power status > /dev/null
        if [[ $? != 0 ]]; then
                nfailed=$((nfailed + 1))
                echo "  Failed on ${sname}"
        fi
done
if [[ ${nfailed} == 0 ]]; then
        echo "  OK"
fi


# Part 3 - The ZCU102 boards

snames=$(grep -e zcu -e snap ${filename} | awk '{print $2}' | sort)

echo "Testing ZCU102 ping-ability"
nfailed=0
for sname in ${snames}; do
        ping -c1 ${sname} > /dev/null 2> /dev/null
        if [[ $? != 0 ]]; then
                nfailed=$((nfailed + 1))
                echo "  Failed on ${sname}"
        fi
done
if [[ ${nfailed} == 0 ]]; then
        echo "  OK"
fi
