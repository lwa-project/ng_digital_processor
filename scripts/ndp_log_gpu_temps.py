#!/usr/bin/env python3

import os
import sys
sys.path.append('/usr/local/bin')

import time
import subprocess

from ndp import NdpConfig
from ndp.NdpHelpers import find_script_pids


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage:", sys.argv[0], "config_file")
        sys.exit(-1)

    pids = find_script_pids(__file__)
    pids = list(filter(lambda x: x != os.getpid(), pids))
    if len(pids) > 0:
        print("Found another %s process running, aborting" % os.path.basename(__file__))
        sys.exit(1)

    config_filename = sys.argv[1]
    config = NdpConfig.parse_config_file(config_filename)
    
    nvidia_cmd = ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits']
    
    data = [time.time(),]
    
    # Head node (local)
    try:
        subdata = []
        output = subprocess.check_output(nvidia_cmd, text=True)
        for line in output.split('\n')[:-1]:
            subdata.append(float(line))
        data.extend(subdata)
    except Exception as e:
        print("WARNING: head node -> %s" % (e,))
        data.extend([-99.0 for i in range(2)])
        
    # Compute servers (remote)
    for server in config['host']['servers']:
        cmd = ['ssh', server] + nvidia_cmd
        
        try:
            subdata = []
            output = subprocess.check_output(cmd, text=True)
            for line in output.split('\n')[:-1]:
                subdata.append(float(line))
            data.extend(subdata)
        except Exception as e:
            print("WARNING: %s -> %s" % (server, e))
            data.extend([-99.0 for i in range(2)])
            
    filename = config['log']['files']['server_temps']
    with open(filename, 'a') as fh:
        fh.write(','.join([str(v) for v in data]))
        fh.write('\n')
