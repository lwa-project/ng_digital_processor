# -*- coding: utf-8 -*-

"""
Disk, CPU, and GPU device monitoring

TODO: `impmitool sensor list` shows there are also thermal-region and PSU temps available
"""

import os
import time
import numpy as np
# Available at https://pypi.python.org/pypi/nvidia-ml-py/
try:
    from pynvml import *
    nvmlInit()
    import atexit
    atexit.register(nvmlShutdown)
except ImportError:
    pass

class DiskDevice(object):
    def __init__(self, path):
        self.path = path
    def id(self):
        return self.path
    def usage(self):
        statvfs = os.statvfs(self.path)
        total_bytes = statvfs.f_frsize * statvfs.f_blocks
        free_bytes  = statvfs.f_frsize * statvfs.f_bfree
        avail_bytes = statvfs.f_frsize * statvfs.f_bavail
        # Note: This matches what df does
        used_bytes = total_bytes - free_bytes
        total_avail_bytes = used_bytes + avail_bytes
        return (total_avail_bytes, used_bytes)

class CPUDevice(object):
    def __init__(self, socket=0):
        self.socket = socket
    def id(self):
        return self.socket
    def temperature(self):
        filename = "/sys/class/thermal/thermal_zone%i/temp" % self.socket
        try:
            with open(filename, 'r') as f:
                contents = f.read()
            return float(contents) / 1000.
        except IOError:
            return float('nan')

class GPUSystem(object):
    def __init__(self):
        #nvmlInit()
        pass
    def __del__(self):
        #nvmlShutdown()
        pass
    def driver_version(self):
        return nvmlSystemGetDriverVersion()
    def device_count(self):
        return nvmlDeviceGetCount()
    #def device(self, idx):
    #	return NVMLDevice(idx)
    def devices(self):
        return [GPUDevice(i) for i in range(self.device_count())]

class GPUDevice(object):
    def __init__(self, idx):
        self.idx = idx
        if isinstance(idx, int):
            self.handle = nvmlDeviceGetHandleByIndex(idx)
        else:
            self.handle = nvmlDeviceGetHandleByPciBusId(idx)
    def id(self):
        return self.idx
    def name(self):
        return nvmlDeviceGetName(self.handle)
    def memory_info(self):
        """Returned object provides .total, .free and .used in bytes"""
        return nvmlDeviceGetMemoryInfo(self.handle)
    def temperature(self):
        """Returns degrees Celcius"""
        return nvmlDeviceGetTemperature(self.handle,
                                        NVML_TEMPERATURE_GPU)
    def temperature_threshold(self):
        """Returns degrees Celcius"""
        return nvmlDeviceGetTemperatureThreshold(self.handle,
                NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
    def fan_speed(self):
        """Returns percentage"""
        return nvmlDeviceGetFanSpeed(self.handle)
