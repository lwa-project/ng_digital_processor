#!/bin/bash

# Simple script to test for various packages, utilties, and Python modules
#
# Usage: ./test_softare_install.sh <hostfile>

filename="../hosts"
if [[ "$1" != "" ]]; then
        filename=$1
fi

if [[ ! -f ${filename} ]]; then
	echo "Host file '${filename}' does not exist"
	exit
fi

snames=$(grep ndp ${filename} | grep -v data | grep -v ipmi | awk '{print $2}' | sort)


# Part 1 - Standard distribution packages

pkgs=("numactl" "libnuma-dev" "hwloc-nox" "libhwloc-dev" "iptables" "sshpass" "*-ctags" "git" "make" "cmake" "rsync" "nfs-kernel-server" "nfs-common")

echo "Testing for standard packages"
nfailed=0
for sname in ${snames}; do
	for pkg in ${pkgs[@]}; do
		if [[ "${pkg}" == "iptables" && "${sname}" != "ndp0" ]]; then
			# iptables is only needed on the headnode
			continue
		fi
		if [[ "${pkg}" == "sshpass" && "${sname}" != "ndp0" ]]; then
			# sshpass is only needed on the headnode
			continue
		fi
		if [[ "${pkg}" == "nfs-kernel-server" && "${sname}" != "ndp0" ]]; then
			# nfs-kernel-server is only needed on the headnode
			continue
		fi
		
		found=$(timeout 10 ssh ndp@${sname} "dpkg -l ${pkg} | grep -e '^ii'")
		if [[ $? != 0 ]]; then
			nfailed=$((nfailed + 1))
	    echo "  Cannot find ${pkg} on ${sname}"
		else
			if [[ "${found}" == "" ]]; then
				nfailed=$((nfailed + 1))
				echo "  Cannot find ${pkg} on ${sname}"
			fi
		fi
	done
done
if [[ $nfailed == 0 ]]; then
	echo "  OK"
fi


# Part 2 - OFED-specific packages

pkgs=("libibverbs-dev" "librdmacm-dev" "mlnx-ofed-kernel-dkms" "ibverbs-utils")

echo "Testing for OFED packages"
nfailed=0
for sname in ${snames}; do
	for pkg in ${pkgs[@]}; do
		nfailed_host=0
		found=$(timeout 10 ssh ndp@${sname} "dpkg -l ${pkg} | grep -e '^ii'")
		if [[ "${found}" == "" ]]; then
			nfailed_host=$((nfailed_host + 1))
			echo "  Cannot find ${pkg} on ${sname}"
	  fi
	done
	
	if [[ $nfailed_host == 0 ]]; then
		nibv=$(ibv_devices | wc -l)
		nibv=$(($ibv - 2))
		if [[ $nibv == 0 ]]; then
			nfailed_host=$(($nfailed_host + 1))
			echo "  ibv_devices list not capable NICs on ${sname}"
		fi
	fi
	nfailed=$((nfailed + nfailed_host))
done
if [[ $nfailed == 0 ]]; then
	echo "  OK"
fi


# Part 3 - CUDA and some GPUs

prgs=("nvidia-smi")

echo "Testing for CUDA"
nfailed=0
for sname in ${snames}; do
	nfailed_host=0
	for prg in ${prgs[@]}; do
		timeout 10 ssh ndp@${sname} "which ${prg}" > /dev/null
		if [[ $? != 0 ]]; then
			nfailed_host=$((nfailed_host + 1))
			echo "  Cannot find ${prg} on ${sname}"
		fi
	done
	
	if [[ $nfailed_host == 0 ]]; then
		ngpu=$(nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv | wc -l)
		ngpu=$(($ngpu - 1))
		if [[ $ngpu == 0 ]]; then
			nfailed_host=$(($nfailed_host + 1))
			echo "  nvidia-smi lists no GPUs on ${sname}"
		fi
	fi
	nfailed=$((nfailed + nfailed_host))
done
if [[ $nfailed == 0 ]]; then
	echo "  OK"
fi


# Part 4 - Python and some modules

prgs=("python" "python3")

echo "Testing for Python"
nfailed=0
for sname in ${snames}; do
	for prg in ${prgs[@]}; do
		which ${prg} > /dev/null
		if [[ $? != 0 ]]; then
			nfailed=$((nfailed + 1))
			echo "  Cannot find ${prg} on ${sname}"
		fi
	done
done
if [[ $nfailed == 0 ]]; then
	echo "  OK"
fi

mods=("numpy" "scipy" "simplejson" "ctypesgen" "zmq" "lwa_f" "yaml" "progressbar" "termcolor" "bifrost" "serial")

echo "Testing for Python modules"

nfailed=0
for sname in ${snames}; do
	for mod in ${mods[@]}; do
		python3 -c "import ${mod}" 2> /dev/null
		if [[ $? != 0 ]]; then
			nfailed=$((nfailed + 1))
			echo "  Cannot find ${mod} on ${sname}"
		fi
	done
done
if [[ $nfailed == 0 ]]; then
	echo "  OK"
fi
