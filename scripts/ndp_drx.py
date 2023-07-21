#! /usr/bin/env python3

import sys
import signal
import logging
import time
import os
import argparse
import threading
import socket
import datetime

__version__    = "1.0"
__date__       = '$LastChangedDate: 2020-25-11$'
__author__     = "Jack Hickish, Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = ""
__credits__    = ["Jack Hickish", "Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jack Hickish"
__email__      = "jack@realtimeradio.co.uk"
__status__     = "Development"

class CoreList(list):
    """
    A dumb class to catch pop-ing too many cores and print an error
    """
    def pop(self, i):
        try:
            return list.pop(self, i)
        except IndexError:
            print("Ran out of CPU cores to use!")
            exit()

def build_pipeline(args):
    from bifrost.address import Address
    from bifrost.udp_socket import UDPSocket
    from bifrost.ring import Ring
    # Blocks
    from lwa352_pipeline.blocks.block_base import Block
    #from lwa352_pipeline.blocks.corr_block import Corr
    from ndp.corr_block import Corr
    from lwa352_pipeline.blocks.dummy_source_block import DummySource
    #from lwa352_pipeline.blocks.corr_acc_block import CorrAcc
    from ndp.corr_acc_block import CorrAcc
    #from lwa352_pipeline.blocks.corr_output_full_block import CorrOutputFull
    from ndp.corr_output_full_block import CorrOutputFull
    from lwa352_pipeline.blocks.copy_block import Copy
    from lwa352_pipeline.blocks.capture_block import Capture
    from lwa352_pipeline.blocks.beamform_block import Beamform
    from lwa352_pipeline.blocks.beamform_vlbi_output_block import BeamformVlbiOutput
    from lwa352_pipeline.blocks.triggered_dump_block import TriggeredDump

    if args.useetcd:
        import etcd3 as etcd
        etcd_client = etcd.client(args.etcdhost)
    else:
        etcd_client = None

    # Set the pipeline ID
    Block.set_id(args.pipelineid)

    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    if args.logfile is None:
        logHandler = logging.StreamHandler(sys.stdout)
    else:
        logHandler = logging.FileHandler(args.logfile)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)
    verbosity = args.verbose
    if   verbosity >  0: log.setLevel(logging.DEBUG)
    elif verbosity == 0: log.setLevel(logging.INFO)
    elif verbosity <  0: log.setLevel(logging.WARNING)
    
    short_date = ' '.join(__date__.split()[1:4])
    log.info("Starting %s with PID %i", sys.argv[0], os.getpid())
    log.info("Cmdline args: \"%s\"", ' '.join(sys.argv[1:]))
    log.info("Version:      %s", __version__)
    log.info("Last changed: %s", short_date)
    log.info("Log file:     %s", args.logfile)
    
    ops = []
    shutdown_event = threading.Event()
    def handle_signal_terminate(signum, frame):
        SIGNAL_NAMES = dict((k, v) for v, k in \
                            reversed(sorted(signal.__dict__.items()))
                            if v.startswith('SIG') and \
                            not v.startswith('SIG_'))
        log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
        try:
            ops[0].shutdown()
        except IndexError:
            pass
        shutdown_event.set()
    for sig in [signal.SIGHUP,
                signal.SIGINT,
                signal.SIGQUIT,
                signal.SIGTERM,
                signal.SIGTSTP]:
        signal.signal(sig, handle_signal_terminate)
    
    
    hostname = socket.gethostname()
    try:
        server_idx = hostname.split('.', 1)[0].replace('lxdlwagpu', '')
        server_idx = int(server_idx, 10)
    except (AttributeError, ValueError):
        server_idx = 1 # HACK to allow testing on head node "adp"
    # TODO: Is there a way to know how many pipelines to expect per server?
    pipeline_idx = server_idx - 1
    log.info("Hostname:     %s", hostname)
    log.info("Server index: %i", server_idx)
    log.info("Pipeline index: %i", args.pipelineid)
    log.info("Global index: %i", pipeline_idx)
    
    capture_ring = Ring(name="capture", space='system')
    gpu_input_ring = Ring(name="gpu-input", space='cuda')
    bf_output_ring = Ring(name="bf-output", space='cuda')
    bf_acc_output_ring = Ring(name="bf-acc-output", space='system')
    corr_output_ring = Ring(name="corr-output", space='cuda')
    corr_slow_output_ring = Ring(name="corr-slow-output", space='cuda_host')
    
    trigger_capture_ring = Ring(name="trigger_capture", space='cuda_host')
    
    # TODO:  Figure out what to do with this resize
    #GSIZE = 480#1200
    NBEAM = 4
    CHAN_PER_PACKET = 96
    NPIPELINE = 2
    NSNAP = 2
    NETGSIZE = 96*2
    NET_NGULP = 5*2 # Number of Net block gulps collated in the first copy
    GSIZE = 480
    GPU_NGULP = 2 # Number of GSIZE gulps in a contiguous GPU memory block
    nstand = 64
    npol = 2
    ninput_per_snap = nstand*npol // NSNAP
    packet_buf_size = ninput_per_snap * CHAN_PER_PACKET + 128
    nchan = args.nchan
    system_nchan = nchan * NPIPELINE

    assert ((NET_NGULP*NETGSIZE) % GSIZE == 0), "GSIZE must be a multiple of NETGSIZE*NET_NGULP"

    cores = CoreList(map(int, args.cores.split(',')))
    
    nfreqblocks = nchan // CHAN_PER_PACKET
    if not args.fakesource:
        print("binding input to %s:%d" %(args.ip, args.port))
        iaddr = Address(args.ip, args.port)
        isock = UDPSocket()
        isock.bind(iaddr)
        isock.timeout = 0.5
        ops.append(Capture(log, fmt="snap2", sock=isock, ring=capture_ring,
                           nsrc=NSNAP*nfreqblocks, src0=0, max_payload_size=packet_buf_size,
                           buffer_ntime=NETGSIZE, slot_ntime=NET_NGULP*NETGSIZE*16,
                           core=cores.pop(0), nstand=nstand, npol=npol, system_nchan=system_nchan,
                           utc_start=datetime.datetime.now(), ibverbs=True))
    else:
        print('Using dummy source...')
        ops.append(DummySource(log, oring=capture_ring, ntime_gulp=NETGSIZE*NET_NGULP, core=cores.pop(0),
                   skip_write=args.nodata, target_throughput=args.target_throughput,
                   nstand=nstand, nchan=nchan, npol=npol, testfile=args.testdatain))

    # capture_ring -> triggered buffer
    ops.append(Copy(log, iring=capture_ring, oring=trigger_capture_ring, ntime_gulp=NETGSIZE,
                      nbyte_per_time=nchan*npol*nstand, buffer_multiplier=GPU_NGULP*NET_NGULP,
                      core=cores.pop(0), guarantee=True, gpu=-1, buf_size_gbytes=args.bufgbytes))

    ops.append(TriggeredDump(log, iring=trigger_capture_ring, ntime_gulp=GSIZE,
                      nbyte_per_time=nchan*npol*nstand,
                      core=cores.pop(0), guarantee=True,
                      etcd_client=etcd_client))

    ops.append(Copy(log, iring=trigger_capture_ring, oring=gpu_input_ring, ntime_gulp=GPU_NGULP*GSIZE,
                      nbyte_per_time=nchan*npol*nstand,
                      core=cores.pop(0), guarantee=True, gpu=args.gpu))

    ops.append(Corr(log, iring=gpu_input_ring, oring=corr_output_ring, ntime_gulp=GSIZE,
                      nchan=nchan, npol=npol, nstand=nstand,
                      core=cores.pop(0), guarantee=True, acc_len=2400, gpu=args.gpu,
                      etcd_client=etcd_client, autostartat=2400*8))

    ops.append(CorrAcc(log, iring=corr_output_ring, oring=corr_slow_output_ring,
                      core=cores.pop(0), guarantee=True, gpu=args.gpu, etcd_client=etcd_client,
                      nchan=nchan, npol=npol, nstand=nstand,
                      acc_len=args.corr_acc_len,
                      autostartat=2400*32*2))

    ops.append(CorrOutputFull(log, iring=corr_slow_output_ring,
                      core=cores.pop(0), guarantee=True, etcd_client=etcd_client,
                      nchan=nchan, npol=npol, nstand=nstand,
                      pipeline_idx=pipeline_idx,
                      npipeline=args.cor_npipeline))

    ops.append(Beamform(log, iring=gpu_input_ring, oring=bf_output_ring, ntime_gulp=GPU_NGULP*GSIZE,
                      nchan=nchan, nbeam=NBEAM*2, ninput=nstand*npol,
                      core=cores.pop(0), guarantee=True, gpu=args.gpu, ntime_sum=None,
                      etcd_client=etcd_client))

    cores.pop(0)
    ops.append(BeamformVlbiOutput(log, iring=bf_output_ring, ntime_gulp=GPU_NGULP*GSIZE,
                                  pipeline_idx=pipeline_idx, core=cores.pop(0),
                                  guarantee=True, etcd_client=etcd_client))

    threads = [threading.Thread(target=op.main) for op in ops]
    
    log.info("Launching %i thread(s)", len(threads))
    try:
        for thread in threads:
            #thread.daemon = True
            thread.start()
        while not shutdown_event.is_set():
            signal.pause()
        log.info("Shutdown, waiting for threads to join")
        for thread in threads:
            thread.join()
        log.info("All done")
        return 0
    except:
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LWA-NA Correlator-Beamformer Pipeline',
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--logfile', default=None,
                        help='Specify log file')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity')
    parser.add_argument('--nchan', type=int, default=96*16,
                        help='Number of frequency channels in the pipeline')
    parser.add_argument('--fakesource', action='store_true',
                        help='Use a dummy source for testing')
    parser.add_argument('--nodata', action='store_true',
                        help='Don\'t generate data in the dummy source (faster)')
    parser.add_argument('--testdatain', type=str, default=None,
                        help='Path to input test data file')
    parser.add_argument('-a', '--corr_acc_len', type=int, default=240000,
                        help='Number of accumulations to start accumulating in the slow correlator')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Which GPU device to use')
    parser.add_argument('--pipelineid', type=int, default=0,
                        help='Pipeline ID. Useful if you are running multiple pipelines on a single machine')
    parser.add_argument('-c', '--cores', default='0,1,2,3,4,5,6,7,8,9',
                        help='Comma-separated list of CPU cores to use')
    parser.add_argument('--useetcd', action='store_true',
                        help='Use etcd control/monitoring server')
    parser.add_argument('--etcdhost', default='etcdhost',
                        help='Host serving etcd functionality')
    parser.add_argument('--ip', default='192.168.40.11',
                        help='IP address to which to bind')
    parser.add_argument('--port', type=int, default=10000,
                        help='UDP port to which to bind')
    parser.add_argument('--bufgbytes', type=int, default=8,
                        help='Number of GBytes to buffer for transient buffering')
    parser.add_argument('--cor_npipeline', type=int, default=1,
                        help='Number of pipelines per COR packet output stream')
    parser.add_argument('--target_throughput', type=float, default='1000.0',
                        help='Target throughput when using --fakesource')
    args = parser.parse_args()

    build_pipeline(args)