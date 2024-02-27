#!/usr/bin/env python3
"""
NDP server monitor service
To be run on NDP servers

Responds to requests from the NDP main control script for:
  SOFTWARE version
  STAT code and INFO string
"""

__version__    = "0.1"
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2024, The Long Wavelength Array Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

# from StoppableThread import StoppableThread
from threading import Thread, Event as ThreadEvent
import zmq
import json
import platform
from ndp.DeviceMonitor import DiskDevice, GPUSystem
from ndp.PipelineMonitor import BifrostPipelines

class ServerMonitor(object):
    def __init__(self, disk_ids=[]):
        self.gpu_system  = GPUSystem()
        self.disks = [DiskDevice(path) for path in disk_ids]
        self.pipelines = BifrostPipelines()
    def get_disk_ids(self):
        return [disk.path for disk in self.disks]
    def get_disk_usages(self):
        return [disk.usage() for disk in self.disks]
    def get_pipelines(self):
        return self.pipelines.pipeline_pids()
    def get_pipeline_data_rates(self):
        return [(p.rx_rate(), p.rx_loss()) for p in self.pipelines.pipelines()]

class NdpServerMonitor(object):
    def __init__(self, config, log, monitor, timeout=0.1):
        self.config  = config
        self.log     = log
        self.monitor = monitor
        self.timeout = timeout
    def request_stop(self):
        try:
            self.event.set()
        except AttributeError:
            pass
    def stop_requested(self):
        try:
            return self.event.is_set()
        except AttributeError:
            return True
    def run(self):
        self.thread = Thread(target=self._run, daemon=True)
        self.event = ThreadEvent()
        self.thread.start()
        self.thread.join()
    def _run(self):
        self.zmqctx  = zmq.Context()
        self.sock    = self.zmqctx.socket(zmq.REP)
        addr = "tcp://%s:%i" % (self.config['mcs']['server']['local_host'],
                                self.config['mcs']['server']['local_port']+1)
        self.log.info("Binding socket to: %s", addr)
        self.sock.bind(addr)
        self.sock.RCVTIMEO = int(self.timeout*1000)
        
        self.log.info("Listening for requests")
        while not self.stop_requested():
            self.log.debug("Waiting for requests")
            try:
                req = self.sock.recv()
                self.log.debug("Received request: %s", req)
                try:
                    req = req.decode()
                    reply = self._process_request(req)
                except Exception as e:
                    reply = {'status': -500,
                             'info':   "Internal server error: %s"%e,
                             'data':   None}
                self.log.debug("Replying with: %s", json.dumps(reply))
                self.sock.send_json(reply)
            except zmq.error.Again:
                pass
            except zmq.error.ZMQError: # For "Interrupted system call"
                pass
                
    def _process_request(self, req):
        status = 0
        info   = 'OK'
        data   = None
        if req == 'PNG':
            pass
        elif req == 'STAT':
            pname = self.monitor.get_pipelines()
            pstat = self.monitor.get_pipeline_data_rates()
            data = ''
            for n,r in zip(pname, pstat):
                r,l = r
                data += f"{n}: RX @ {r/1024**3:.2f GiB/s} with {l:.1%} inst. packet loss, "
            data = data[:-2]
        elif req == 'INFO':
            dname = self.monitor.get_disk_ids()
            dusage = self.monitor.get_disk_usages()
            data = ''
            for n,u in zip(dname, dusage):
                t,u = u
                data += f"{n}: using {u/1024**3:.1f} GiB of {t/1024**3:.1f} GiB, "
            data = data[:-2]
        elif req == 'SOFTWARE':
            # TODO: Get pipeline version etc.
            kernel_version = platform.uname()[2]
            data = ("krnl:%s,ndp_srv_mon:%s,gpu_drv:%s" %
            (kernel_version,
             __version__,
             self.monitor.gpu_system.driver_version()))
        else:
            status = -1
            info   = "Unknown MIB entry: %s" % req
            self.log.error(info)
        return {'status': status,
                'info':   info,
                'data':   data}

from ndp import NdpConfig
from ndp import MCS2
import logging
from logging.handlers import TimedRotatingFileHandler
import time
import os
import signal

def main(argv):
    import sys
    if len(sys.argv) <= 1:
        print("Usage:", sys.argv[0], "config_file")
        sys.exit(-1)
    config_filename = sys.argv[1]
    config = NdpConfig.parse_config_file(config_filename)

    # TODO: Try to encapsulate this in something simple in NdpLogging
    log = logging.getLogger(__name__)
    logFormat = logging.Formatter(config['log']['msg_format'],
                                  datefmt=config['log']['date_format'])
    logFormat.converter = time.gmtime
    logHandler = logging.StreamHandler(sys.stdout)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)
    log.setLevel(logging.INFO)

    log.info("Starting %s with PID %i", argv[0], os.getpid())
    log.info("Cmdline args: \"%s\"", ' '.join(argv[1:]))
    log.info("Version:      %s", __version__)
    log.info("Current MJD:  %f", MCS2.slot2mjd())
    log.info("Current MPM:  %i", MCS2.slot2mpm())
    log.info('All dates and times are in UTC unless otherwise noted')

    log.debug("Creating server monitor")
    monitor = ServerMonitor(config['server']['disk_ids'])

    log.debug("Creating NDP server monitor")
    ndpmon = NdpServerMonitor(config, log, monitor)

    def handle_signal_terminate(signum, frame):
        SIGNAL_NAMES = dict((k, v) for v, k in \
                            reversed(sorted(signal.__dict__.items()))
                            if v.startswith('SIG') and \
                            not v.startswith('SIG_'))
        log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
        ndpmon.request_stop()
    log.debug("Setting signal handlers")
    signal.signal(signal.SIGHUP,  handle_signal_terminate)
    signal.signal(signal.SIGINT,  handle_signal_terminate)
    signal.signal(signal.SIGQUIT, handle_signal_terminate)
    signal.signal(signal.SIGTERM, handle_signal_terminate)
    signal.signal(signal.SIGTSTP, handle_signal_terminate)

    log.debug("Running NDP server monitor")
    ndpmon.run()

    log.info("All done, exiting")
    sys.exit(0)
	
if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
