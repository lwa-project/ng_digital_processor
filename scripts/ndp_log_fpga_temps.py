#!/usr/bin/env python3

import os
import sys
sys.path.append('/usr/local/bin')

import time

from ndp import NdpConfig
from ndp.NdpFpga import check as fpga_check
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
    
    out = [time.time(),]
    for s in config['host']['fpgas']:
        try:
            temps = [-1.,]
            ret = fpga_check(s, 'temperature')
            temps = [round(ret.get('fpga', -99.), 2),]
        except Exception as e:
            print("WARNING: Failed to poll %s: %s" % (s, str(e)))
            
        for t in temps:
            out.append( t )
            
    filename = config['log']['files']['fpga_temps']
    with open(filename, 'a') as fh:
        fh.write(','.join([str(v) for v in out]))
        fh.write('\n')
