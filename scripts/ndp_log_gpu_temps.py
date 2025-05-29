#!/usr/bin/env python3

import os
import sys
import time
import subprocess


if __name__ == "__main__":
    data = [time.time(),]
    for server in range(0, 4+1):
        cmd = []
        if server != 0:
            cmd.extend(['ssh', 'ndp%i' % server])
        cmd.extend(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'])
        
        try:
            subdata = []
            output = subprocess.check_output(cmd)
            output = output.decode()
            for line in output.split('\n')[:-1]:
                value = float(line)
                subdata.append(value)
            data.extend(subdata)
            
        except Exception as e:
            print("WARNING: %i -> %s" % (server, e))
            data.extend([-99.0 for i in range(2)])

    with open('/home/ndp/log/gpu.txt', 'a') as fh:
        fh.write(','.join([str(v) for v in data]))
        fh.write('\n')
