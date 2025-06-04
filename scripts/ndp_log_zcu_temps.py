#!/usr/bin/env python3

import sys
sys.path.append('/usr/local/bin')

import time

from ndp import NdpConfig
from ndp.FileLock import FileLock

from lwa_f import zcu102_fengine


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage:", sys.argv[0], "config_file")
        sys.exit(-1)
        
    config_filename = sys.argv[1]
    config = NdpConfig.parse_config_file(config_filename)
    
    out = [time.time(),]
    for s in config['host']['zcus']:
        zcu = zcu102_fengine.ZCU102Fengine(s,
                                           username=config['zcu']['username'],
                                           password=config['zcu']['password'])
        lock = FileLock(f"/dev/shm/{s}_access")
        
        try:
            temps = [-1.,]
            with lock:
                if zcu.is_connected:
                    if zcu.fpga.is_programmed():
                        summary, flags = zcu.fpga.get_status()
                        temps = [summary['temp'],]
        except Exception as e:
            print("WARNING: Failed to poll %s: %s" % (s, str(e)))
            
        for t in temps:
            out.append( t )
            
    with open('/home/ndp/log/snap.txt', 'a') as fh:
        fh.write(','.join([str(v) for v in out]))
        fh.write('\n')
