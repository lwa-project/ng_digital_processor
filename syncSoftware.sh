#!/bin/bash

#
# Host validation
#

if [ `hostname` != "ndp" ]; then
	echo "This script must be run on the head node"
	exit 1
fi

#
# Argument parsing
#

DO_CONFIG=1
DO_SOFTWARE=1
DO_UPSTART=1
DO_RESTART=0
DO_QUERY=0
while [[ $# -gt 0 ]]; do
	key="${1}"
	case ${key} in
		-h|--help)
			echo "syncSoftware.py - Script to help get the NDP software in sync"
			echo "                  across the various nodes."
			echo ""
			echo "Usage:"
			echo "sudo ./syncSoftware.py [OPTIONS]"
			echo ""
			echo "Options:"
			echo "-h,--help            Show this help message"
			echo "-c,--config-only     Only update the configuration file and restart the NDP services"
			echo "-s,--software-only   Only update the NDP software and restart the NDP services"
			echo "-u,--upstart-only    Only update the NDP systemd service definitions"
			echo "-r,--restart         Rrestart the NDP services after an update"
			echo "-o,--restart-only    Do not udpdate, only restart the NDP services"
			echo "-q,--query           Query the status of the NDP services"
			exit 0
			;;
		-c|--config-only)
			DO_CONFIG=1
			DO_SOFTWARE=0
			DO_UPSTART=0
			DO_QUERY=0
			;;
		-s|--software-only)
			DO_CONFIG=0
			DO_SOFTWARE=1
			DO_UPSTART=0
			DO_QUERY=0
			;;
		-u|--upstart-only)
			DO_CONFIG=0
			DO_SOFTWARE=0
			DO_UPSTART=1
			DO_QUERY=0
			;;
		-r|--restart)
			DO_RESTART=1
			DO_QUERY=0
			;;
		-0|--restart-only)
			DO_CONFIG=0
			DO_SOFTWARE=0
			DO_UPSTART=0
			DO_RESTART=1
			DO_QUERY=0
			;;
		-q|--query)
			DO_CONFIG=0
			DO_SOFTWARE=0
			DO_UPSTART=0
			DO_RESTART=0
			DO_QUERY=1
			;;
		*)
		;;
	esac
	shift
done

#
# Permission validation
#

if [ `whoami` != "root" ]; then
	echo "This script must be run with superuser privileges"
	exit 2
fi

#
# TCC setup function
#

build_tcc() {
	## Path setup for CUDA
	PATH=$PATH:/usr/local/cuda/bin
	
	## The Bifrost source to build against
	BIFROST_PATH=`grep -e "BIFROST_INCLUDE" config/servers/ndp-drx-0.service | sed -e 's/.*=//g;s/\/src\/bifrost//g;'`
	
	cdir=`pwd`
	cd bifrost_tcc_wrapper/tensor-core-correlator
	make clean
	make
	
	cd ${cdir}
	cd bifrost_tcc_wrapper/bifrost
	python3 make_bifrost_plugin.py -b ${BIFROST_PATH} btcc.cu
}

#
# Configuration
#

if [ "${DO_CONFIG}" == "1" ]; then
	SRC_PATH=/home/ndp/ng_digital_processor/config
	DST_PATH=/usr/local/share/ndp
	
	for node in `seq 0 2`; do
		rsync -e ssh -avHL ${SRC_PATH}/ndp_config.json ndp${node}:${DST_PATH}/
		rsync -e ssh -avH ${SRC_PATH}/equalizer.txt ndp${node}:${DST_PATH}/
	done
fi


#
# Software
#

if [ "${DO_SOFTWARE}" == "1" ]; then
	SRC_PATH=/home/ndp/ng_digital_processor/scripts
	TCC_PATH=/home/ndp/ng_digital_processor/bifrost_tcc_wrapper/bifrost
	DST_PATH=/usr/local/bin
	
	build_tcc
	
	for node in `seq 0 2`; do
		if [ "${node}" == "0" ]; then
			rsync -e ssh -avH ${SRC_PATH}/ndp ${SRC_PATH}/ndp_control.py ${SRC_PATH}/ndp_tengine.py ${SRC_PATH}/ndp_enable_triggering.py ndp${node}:${DST_PATH}/
		else
			rsync -e ssh -avH ${SRC_PATH}/ndp ${SRC_PATH}/ndp_drx.py ndp${node}:${DST_PATH}/
			rsync -e ssh -avH ${TCC_PATH}/bt*.py ${TCC_PATH}/*.so* ndp${node}:${DST_PATH}/
		fi
	done
fi


#
# Upstart
#

if [ "${DO_UPSTART}" == "1" ]; then
	SRC_PATH=/home/ndp/ng_digital_processor/config
	DST_PATH=/etc/systemd/system/
	
	for node in `seq 0 2`; do
		if [ "${node}" == "0" ]; then
			rsync -e ssh -avH ${SRC_PATH}/headnode/ndp-*.service ndp${node}:${DST_PATH}/
		elif [ "${node}" == "1" ]; then
			rsync -e ssh -avH ${SRC_PATH}/servers/ndp-drx-[01].service ndp${node}:${DST_PATH}/
		else
			rsync -e ssh -avH ${SRC_PATH}/servers/ndp-drx-[23].service ndp${node}:${DST_PATH}/
		fi
		ssh ndp${node} "systemctl daemon-reload"
	done
fi

#
# Restart
#

if [ "${DO_RESTART}" == "1" ]; then
	for node in `seq 0 6`; do
		if [ "${node}" == "0" ]; then
			ssh ndp${node} "restart ndp-control && restart ndp-tengine-0 && restart ndp-tengine-1 && restart ndp-tengine-2 && restart ndp-tengine-3"
		elif [ "${node}" == "1" ]; then
			ssh ndp${node} "restart ndp-drx-0 && restart ndp-drx-1"
		else
			ssh ndp${node} "restart ndp-drx-2 && restart ndp-drx-3"
		fi
	done
fi

#
# Query
#

if [ "${DO_QUERY}" == "1" ]; then
	for node in `seq 0 2`; do
		if [ "${node}" == "0" ]; then
			ssh ndp${node} "status ndp-control && status ndp-tengine-0 && status ndp-tengine-1 && status ndp-tengine-2 && status ndp-tengine-3"
		elif [ "${node}" == "1" ]; then
			ssh ndp${node} "status ndp-tbn && status ndp-drx-0 && status ndp-drx-1"
		else
			ssh ndp${node} "status ndp-tbn && status ndp-drx-2 && status ndp-drx-3"
		fi
	done
fi
