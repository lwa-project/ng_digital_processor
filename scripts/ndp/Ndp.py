
from .NdpCommon  import *
from .NdpConfig  import *
from .NdpLogging import *

from . import MCS2
from .PipelineMonitor import BifrostPipelines
from .ConsumerThread import ConsumerThread
from .SequenceDict import SequenceDict
from .ThreadPool import ThreadPool
from .ThreadPool import ObjectPool
from .iptools    import *

from . import ISC

from lwa_f import snap2_fengine
from .FileLock import FileLock

from queue import Queue
import numpy as np
import time
import math
from collections import defaultdict, OrderedDict
from functools import lru_cache
import logging
import struct
import subprocess
import datetime
import zmq
import threading
import socket # For socket.error
import json
import yaml
import hashlib

__version__    = "0.3"
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2023, The Long Wavelength Array Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

# Global shared resources
#_g_thread_pool = ThreadPool()
_g_zmqctx      = zmq.Context()

def wait_until_utc_sec(utcstr):
    cur_time = datetime.datetime.utcnow().strftime(DATE_FORMAT)
    while cur_time != utcstr:
        time.sleep(0.01)
        cur_time = datetime.datetime.utcnow().strftime(DATE_FORMAT)

class SlotCommandProcessor(object):
    def __init__(self, cmd_code, cmd_parser, exec_delay=2):
        self.cmd_sequence = defaultdict(list)
        self.exec_delay   = exec_delay
        self.cmd_code     = cmd_code
        self.cmd_parser   = cmd_parser
        
    @ISC.logException
    def process_command(self, msg):
        assert( msg.cmd == self.cmd_code )
        exec_slot = msg.slot + self.exec_delay
        self.cmd_sequence[exec_slot].append(self.cmd_parser(msg))
        return 0
        
    @ISC.logException
    def execute_commands(self, slot):
        try:
            cmds = self.cmd_sequence.pop(slot)
        except KeyError:
            return
        return self.execute(cmds)


class DrxCommand(object):
    def __init__(self, msg):
        if isinstance(msg.data, str):
            msg.data = msg.data.encode()
        self.beam, self.tuning, self.freq, self.filt, self.gain, self.subslot \
            = struct.unpack('>BBfBhB', msg.data)
        assert( 1 <= self.beam <= 4 )
        assert( 1 <= self.tuning <= 2 )
        # TODO: Check allowed range of freq
        assert( 0 <= self.filt    <= 8 )
        assert( 0 <= self.gain    <= 15 )
        assert( 0 <= self.subslot <= 99)


class Drx(SlotCommandProcessor):
    def __init__(self, config, log, messenger, servers):
        SlotCommandProcessor.__init__(self, 'DRX', DrxCommand)
        self.config  = config
        self.log     = log
        self.messenger = messenger
        self.servers = servers
        self.nbeam = 4
        self.ntuning = 2
        self.cur_freq = [0]*self.nbeam*self.ntuning
        self.cur_filt = [0]*self.nbeam*self.ntuning
        self.cur_gain = [0]*self.nbeam*self.ntuning
        
    def _reset_state(self):
        for i in range(self.nbeam*self.ntuning):
            self.cur_freq[i] = 0
            self.cur_filt[i] = 0
            self.cur_gain[i] = 0
            
    def tune(self, beam=0, tuning=0, freq=38.00e6, filt=1, gain=1, subslot=0, internal=False):
        ## Convert to the DP frequency scale
        freq = FS * int(freq / FS * 2**32) / 2**32
        
        self.log.info("Tuning DRX %i-%i: freq=%f,filt=%i,gain=%i @ subslot=%i" % (beam, tuning, freq, filt, gain, subslot))
        
        self.messenger.drxConfig(beam, tuning, freq, filt, gain, subslot)
        
        if not internal:
            self.cur_freq[beam*self.ntuning + tuning] = freq
            self.cur_filt[beam*self.ntuning + tuning] = filt
            self.cur_gain[beam*self.ntuning + tuning] = gain
            
        return True
        
    def start(self, beam=0, tuning=0, freq=59.98e6, filt=1, gain=1, subslot=0):
        ## Convert to the DP frequency scale
        freq = FS * int(freq / FS * 2**32) / 2**32
        
        self.log.info("Starting DRX %i-%i data: freq=%f,filt=%i,gain=%i,subslot=%i" % (beam, tuning, freq, filt, gain, subslot))
        
        ret = self.tune(beam=beam, tuning=tuning, freq=freq, filt=filt, gain=gain, subslot=subslot)
        
        return ret
        
    def execute(self, cmds):
        for cmd in cmds:
            # Note: Converts from 1-based to 0-based tuning
            self.start(cmd.beam-1, cmd.tuning-1, cmd.freq, cmd.filt, cmd.gain, cmd.subslot)
            
    def stop(self):
        self.log.info("Stopping DRX data")
        self.cur_freq = [0]*self.nbeam*self.ntuning
        self.cur_filt = [0]*self.nbeam*self.ntuning
        self.cur_gain = [0]*self.nbeam*self.ntuning
        self.log.info("DRX stopped")
        return 0


class TbfCommand(object):
    @ISC.logException
    def __init__(self, msg):
        if isinstance(msg.data, str):
            msg.data = msg.data.encode()
        self.bits, self.trigger, self.samples, self.mask \
            = struct.unpack('>Biiq', msg.data)


class Tbf(SlotCommandProcessor):
    @ISC.logException
    def __init__(self, config, log, messenger, servers):
        SlotCommandProcessor.__init__(self, 'TBF', TbfCommand)
        self.config  = config
        self.log     = log
        self.messenger = messenger
        self.servers = servers
        self.cur_bits = self.cur_trigger = self.cur_samples = self.cur_mask = 0
        
    def _reset_state(self):
        self.cur_bits = self.cur_trigger = self.cur_samples = self.cur_mask = 0
        
    @ISC.logException
    def start(self, bits, trigger, samples, mask):
        self.log.info('Starting TBF: bits=%i, trigger=%i, samples=%i, mask=%i' % (bits, trigger, samples, mask))
        
        self.messenger.trigger(trigger, samples, mask)
        
        return True
        
    @ISC.logException
    def execute(self, cmds):
        for cmd in cmds:
            self.start(cmd.bits, cmd.trigger, cmd.samples, cmd.mask)
            
    def stop(self):
        return False


class BamCommand(object):
    @ISC.logException
    def __init__(self, msg):
        self.beam = struct.unpack('>H', msg.data[0:2])[0]
        self.delays = np.ndarray((512,), dtype='>H', buffer=msg.data[2:1026])
        self.gains = np.ndarray((256,2,2), dtype='>H', buffer=msg.data[1026:3074])
        self.subslot = struct.unpack('>B', msg.data[3074:3075])[0]


class Bam(SlotCommandProcessor):
    @ISC.logException
    def __init__(self, config, log, messenger, servers):
        SlotCommandProcessor.__init__(self, 'BAM', BamCommand)
        self.config  = config
        self.log     = log
        self.messenger = messenger
        self.servers = servers
        self.nbeam = 4
        self.cur_beam = [0]*self.nbeam
        self.cur_delays = [[0 for i in range(512)]]*self.nbeam
        self.cur_gains = [[0 for i in range(1024)]]*self.nbeam
        
    def _reset_state(self):
        for i in range(self.nbeam):
            self.cur_beam[i] = 0
            self.cur_delays[i] = [0 for j in range(512)]
            self.cur_gains[i] = [0 for j in range(1024)]
            
    @ISC.logException
    def start(self, beam, delays, gains, subslot):
        self.log.info("Setting BAM: beam=%i, subslot=%i" % (beam, subslot))
        
        self.messenger.bamConfig(beam, delays, gains, subslot)
        
        return True
        
    @ISC.logException
    def execute(self, cmds):
        for cmd in cmds:
            # Note: Converts from 1-based to 0-based tuning
            self.start(cmd.beam-1, cmd.delays, cmd.gains, cmd.subslot)
            
    def stop(self):
        return False


class CorCommand(object):
    @ISC.logException
    def __init__(self, msg):
        self.navg, self.gain, self.subslot \
            = struct.unpack('>ihB', msg)


class Cor(SlotCommandProcessor):
    @ISC.logException
    def __init__(self, config, log, messenger, servers):
        SlotCommandProcessor.__init__(self, 'COR', CorCommand)
        self.config  = config
        self.log     = log
        self.messenger = messenger
        self.servers = servers
        self.ntuning = 4
        self.cur_navg = [0]*self.ntuning
        self.cur_gain = [0]*self.ntuning
        
    def _reset_state(self):
        for i in range(self.ntuning):
            self.cur_navg[i] = 0
            self.cur_gain[i] = 0
            
    @ISC.logException
    def start(self, navg, gain, subslot):
        self.log.info("Setting COR: navg=%i, gain=%i, subslot=%i" % (navg, gain, subslot))
        
        self.messenger.corConfig(navg, gain, subslot)
        
        return True
        
    @ISC.logException
    def execute(self, cmds):
        for cmd in cmds:
            # Note: Converts from 1-based to 0-based tuning
            self.start(cmd.navg, cmd.gain, cmd.subslot)
            
    def stop(self):
        return False


"""
class FstCommand(object):
    def __init__(self, msg):
        self.index = int(struct.unpack('>h', (msg.data[0:2]))[0])
        self.coefs = np.ndarray((16,32), dtype='>h', buffer=msg.data[2:])
class Fst(object):
    def __init__(self, config, log,
                nupdate_save=5):
        self.config = config
        self.log    = log
        hosts = config['server_hosts']
        ports = config['fst']['control_ports']
        self.addrs = ['tcp://%s:%i'%(hosts[i//2],ports[i%2]) \
                    for i in range(len(hosts)*len(ports))]
        self.socks = ObjectPool([_g_zmqctx.socket(zmq.REQ) \
                                for _ in self.addrs])
        for sock,addr in zip(self.socks,self.addrs):
            try: sock.connect(addr)
            except zmq.error.ZMQError:
                self.log.error("Invalid or non-existent address: %s" %
                                addr)
                # TODO: How to bail out?
        self.exec_delay = 2
        self.cmd_sequence = defaultdict(list)
        self.fir_coefs = SequenceDict(lambda : np.ones((NSTAND,NPOL,
                                                        FIR_NFINE,FIR_NCOEF),
                                                    dtype=np.int16),
                                    maxlen=nupdate_save)
        self.fir_coefs[0][...] = self._load_default_fir_coefs()
        #self.future_pool = FuturePool(len(self.socks))
    def _load_default_fir_coefs(self):
        nfine = self.fir_coefs[-1].shape[-2]
        ncoef = self.fir_coefs[-1].shape[-1]
        fir_coefs = np.fromfile(self.config['fst']['default_coeffs'],
                                dtype='>h').reshape(nfine,ncoef)
        return fir_coefs[None,None,:,:]
    def process_command(self, msg):
        assert( msg.cmd == 'FST' )
        exec_slot = msg.slot + self.exec_delay
        self.cmd_sequence[exec_slot].append(FstCommand(msg))
    def execute_commands(self, slot):
        try:
            cmds = self.cmd_sequence.pop(slot)
        except KeyError:
            return
        # Start with current coefs
        self.fir_coefs[slot][...] = self.fir_coefs.at(slot-1)
        # Merge updates into the set of coefficients
        for cmd in cmds:
            if cmd.index == -1:
                self.fir_coefs[slot][...] = self._load_default_fir_coefs()
            elif cmd.index == 0:
                # Apply same coefs to all inputs
                self.fir_coefs[slot][...] = cmd.coefs[None,None,:,:]
            else:
                stand = (cmd.index-1) // 2
                pol   = (cmd.index-1) % 2
                self.fir_coefs[slot][stand,pol] = cmd.coefs
        self._send_update(slot)
    def get_fir_coefs(self, slot):
        # Access history of updates
        return self.fir_coefs.at(slot)
    def _send_update(self, slot):
        weights = get_freq_domain_filter(self.fir_coefs[slot])
        # weights: [stand,pol,chan] complex64
        weights = weights.transpose(2,0,1)
        # weights: [chan,stand,pol] complex64
        weights /= weights.max() # Normalise to max DC gain of 1.0
        # Send update to pipelines
        # Note: We send all weights to all servers and let them extract
        #         the channels they need, rather than trying to keep
        #         track of which servers have which channels from here.
        # TODO: If msg traffic ever becomes a problem, could probably
        #         use fp16 instead of fp32 for these.
        #hdr  = struct.pack('@iihc', slot, NCHAN, NSTAND, NPOL)
        hdr = json.dumps({'slot':   slot,
                        'nchan':  NCHAN,
                        'nstand': NSTAND,
                        'npol':   NPOL})
        data = weights.astype(np.complex64).tobytes()
        msg  = hdr+data

        self.socks.send_multipart([hdr, data])
        replies = self.socks.recv_json()
        #def send_msg(sock):
        #	sock.send_multipart([hdr, data])
        #	# TODO: Add receive timeout
        #	return sock.recv_json()
        #for sock in self.socks:
        #	self.future_pool.add_task(send_msg, sock)
        #replies = self.future_pool.wait()

        for reply,addr in zip(replies,self.addrs):
            if reply['status'] < 0:
                self.log.error("Gain update failed "
                            "for address %s: (%i) %s" %
                            addr, reply['status'], reply['info'])
"""


# Special response packing functions
def pack_reply_CMD_STAT(slot, cmds):
    ncmd_max = 606
    cmds = cmds[:ncmd_max]
    fmt = '>LH%dL%dB' % (len(cmds), len(cmds))
    responseParts = [slot, len(cmds)]
    responseParts.extend( [cmd[1] for cmd in cmds] )
    responseParts.extend( [cmd[2] for cmd in cmds] )
    return struct.pack(fmt, *responseParts)


def truncate_message(s, n):
    return s if len(s) <= n else s[:n-3] + '...'


def pretty_print_bytes(bytestring):
    return ' '.join(['%02x' % ord(i) for i in bytestring])


# HACK TESTING
#lock = threading.Lock()


class NdpServerMonitorClient(object):
    def __init__(self, config, log, host, timeout=0.1):
        self.config = config
        self.log  = log
        self.host = host
        self.host_ipmi = self.host + "-ipmi"
        self.port = config['mcs']['server']['local_port']
        self.sock = _g_zmqctx.socket(zmq.REQ)
        addr = 'tcp://%s:%i' % (self.host,self.port)
        try: self.sock.connect(addr)
        except zmq.error.ZMQError:
            self.log.error("Invalid or non-existent address: %s" % addr)
        self.sock.SNDTIMEO = int(timeout*1000)
        self.sock.RCVTIMEO = int(timeout*1000)
        
    def read_sensors(self):
        ret = self._ipmi_command('sdr')
        sensors = {}
        for line in ret.split('\n'):
            if '|' not in line:
                continue
            cols = [col.strip() for col in line.split('|')]
            key = cols[0]
            val = cols[1].split()[0]
            sensors[key] = val
        return sensors
        
    @lru_cache(maxsize=4)
    def get_temperatures(self, slot):
        try:
            sensors = self.read_sensors()
            return {key: float(sensors[key])
                    for key in self.config['server']['temperatures']
                    if  key in sensors}
        except:
            return {'error': float('nan')}
            
    @lru_cache(maxsize=4)
    def get_status(self, slot):
        return self._request('STAT')
        
    @lru_cache(maxsize=4)
    def get_info(self, slot):
        return self._request('INFO')
        
    @lru_cache(maxsize=4)
    def get_software(self, slot):
        return self._request('SOFTWARE')
        
    def _request(self, query):
        try:
            self.sock.send(query)
            response = self.sock.recv_json()
        except zmq.error.Again:
            raise RuntimeError("Server '%s' did not respond" % self.host)
        # TODO: Choose a better form of status codes
        if response['status'] == -404:
            raise KeyError
        elif response['status'] < 0:
            raise RuntimeError(response['info'])
        else:
            return response['data']
            
    def get_power_state(self):
        """Returns 'on' or 'off'"""
        return self._ipmi_command("power status").split()[-1]
        
    def do_power(self, op='status'):
        return self._ipmi_command("power "+op)
        
    def _ipmi_command(self, cmd):
        username = self.config['ipmi']['username']
        password = self.config['ipmi']['password']
        #try:
        ret = subprocess.check_output(['ipmitool', '-H', self.host_ipmi,
                                       '-U', username, '-P', password] +
                                       cmd.split())
        return ret.decode()
        
    def stop_tengine(self, beam=0):
        try:
            self._shell_command("systemctl stop ndp-tengine-%i" % beam)
            return True
        except subprocess.CalledProcessError:
            return False
            
    def start_tengine(self, beam=0):
        try:
            self._shell_command("systemctl start ndp-tengine-%i" % beam)
            return True
        except subprocess.CalledProcessError:
            return False
            
    def restart_tengine(self, beam=0):
        self.stop_tengine(beam=beam)
        return self.start_tengine(beam=beam)
        
    def status_tengine(self, beam=0):
        try:
            return self._shell_command("status ndp-tengine-%i" % beam)
        except subprocess.CalledProcessError:
            return "unknown"
            
    def pid_tengine(self, beam=0):
        try:
            pids = self._shell_command("ps aux | grep ndp_tengine | grep -- --beam[=\ ]%i | grep -v grep | awk '{print $2}'" % beam)
            pids = pids.split('\n')[:-1]
            pids = [int(pid, 10) for pid in pids]
            if len(pids) == 0:
                pids = [-1,]
            return pids 
        except subprocess.CalledProcessError:
            return [-1,]
        except ValueError:
            return [-1,]
            
    def stop_drx(self, tuning=0):
        try:
            self._shell_command("systemctl stop ndp-drx-%i" % tuning)
            return True
        except subprocess.CalledProcessError:
            return False
        
    def start_drx(self, tuning=0):
        try:
            self._shell_command("systemctl start ndp-drx-%i" % tuning)
            return True
        except subprocess.CalledProcessError:
            return False
            
    def restart_drx(self, tuning=0):
        self.stop_drx(tuning=tuning)
        return self.start_drx(tuning=tuning)
        
    def status_drx(self, tuning=0):
        try:
            return self._shell_command("status ndp-drx-%i" % tuning)
        except subprocess.CalledProcessError:
            return "unknown"
            
    def pid_drx(self, tuning=0):
        try:
            pids = self._shell_command("ps aux | grep ndp_drx | grep -- --tuning[=\ ]%i | grep -v grep | awk '{print $2}'"  % tuning)
            pids = pids.split('\n')[:-1]
            pids = [int(pid, 10) for pid in pids]
            if len(pids) == 0:
                pids = [-1,]
            return pids 
        except subprocess.CalledProcessError:
            return [-1,]
        except ValueError:
            return [-1,]
            
    def kill_pid(self, pid):
        try:
            self._shell_command("kill -9 %i" % pid)
            return True
        except subprocess.CalledProcessError:
            return False
            
    def _shell_command(self, cmd, timeout=5.):
        #self.log.info("Executing "+cmd+" on "+self.host)
        
        password = self.config['server']['password']
        #self.log.info("RUNNING SSHPASS " + cmd)
        ret = subprocess.check_output(['sshpass', '-p', password,
                                       'ssh', '-o', 'StrictHostKeyChecking=no',
                                       'root@'+self.host,
                                       cmd])
        try:
            ret = ret.decode()
        except AttributeError:
            # Python2 catch
            pass
        #self.log.info("SSHPASS DONE: " + ret)
        #self.log.info("Command executed: "+ret)
        return ret
        
    def can_ssh(self):
        try:
            #with lock:
            ret = self._shell_command('hostname')
            return True
            #except socket.error:
        #except RuntimeError:
        except subprocess.CalledProcessError:
            return False


STAT_SAMP_SIZE = 256

class Snap2MonitorClient(object):
    def __init__(self, config, log, num):
        # Note: num is 1-based index of the snap
        self.config = config
        self.log    = log
        self.num    = num
        self.host   = f"snap{self.num:02d}"
        self.snap   = snap2_fengine.Snap2Fengine(self.host)
        
        self.equalizer_coeffs = None
        try:
            self.equalizer_coeffs = np.loadtxt(self.config['snap']['equalizer_coeffs'])
        except Exception as e:
            self.log.warning("Failed to load equalizer coefficients: %s", str(e))
            
        self.access_lock = FileLock(f"/dev/shm/{self.host}_access")
        
    def unprogram(self, reboot=False):
        with self.access_lock:
            if self.snap.is_connected:
                self.snap.deprogram()
                
    def get_samples(self, slot, stand, pol, nsamps=None):
        with self.access_lock:
            return self.get_samples_all(slot, nsamps)[stand,pol]
            
    @lru_cache(maxsize=4)
    def get_samples_all(self, slot, nsamps=None):
        """Returns an NDArray of shape (stand,pol,sample)"""
        samps = np.zeros((32,2,STAT_SAMP_SIZE))
        with self.access_lock:
            if self.snap.is_connected and self.snap.fpga.is_programmed():
                samps0 = self.snap.adc.get_snapshot_interleaved(0, signed=True, trigger=True)
                samps1 = self.snap.adc.get_snapshot_interleaved(1, signed=True, trigger=False)
                samps = np.vstack([samps0, samps1])
                
        return samps.reshape(32,2,-1)
        
    @lru_cache(maxsize=4)
    def get_temperatures(self, slot):
        # Return a dictionary of temperatures on the Snap2 board
        temp = {'error': float('nan')}
        with self.access_lock:
            if self.snap.is_connected:
                try:
                    summary, flags = self.snap.fpga.get_status()
                    temp = {'fpga': summary['temp']}
                except Exception as e:
                    pass
        return temp
        
    @lru_cache(maxsize=4)
    def get_clock_rate(self):
        # Estimate the FPGA clock rate in MHz
        rate = float('nan')
        with self.access_lock:
            if self.snap.is_connected:
                try:
                    rate = self.snap.fpga.get_fpga_clock()
                except Exception as e:
                    pass
        return rate
        
    def get_tt_of_sync(self, wait_for_sync=True):
        # Return the timetag corresponding to a sync pulse.
        tt = None
        with self.access_lock:
            if self.snap.is_connected and self.snap.fpga.is_programmed():
                tt = self.snap.sync.get_tt_of_sync(wait_for_sync=wait_for_sync)
                tt = tt[0]
        return tt
        
    def program(self):
        # Program with NDP firmware
        
        ## Firmware
        firmware = self.config['snap']['firmware']
        
        # Go!
        success = False
        with self.access_lock:
            if self.snap.is_connected:
                for i in range(self.config['snap']['max_program_attempts']):
                    try:
                        self.snap.program(firmware)
                        sucesss = True
                        break
                    except Exception as e:
                        pass
                        
        return success
        
    def is_programmed(self):
        # Has the FPGA been programmed?
        
        status = False
        with self.access_lock:
            if self.snap.is_connected:
                try:
                    status = self.snap.fpga.is_programmed()
                except Exception as e:
                    pass
        return status
        
    def configure(self):
        # Configure the FPGA and start data flowing
        
        ## Configuration
        ### Overall structure + base F-engine config.
        sconf = {'fengines': {'enable_pfb': not self.config['snap']['bypass_pfb'],
                              'fft_shift': self.config['snap']['fft_shift'],
                              'chans_per_packet': self.config['snap']['nchan_packet'],
                              'adc_clocksource': 0},
                 'xengines': {'arp': {},
                              'chans': {}}}
                 
        ### Toggle FFT shift schedule on/off
        if sconf['fengines']['fft_shift'] in ('', 0):
            del sconf['fengines']['fft_shift']
            
        ### Equalizer coefficints (global)
        if self.equalizer_coeffs is not None:
            sconf['fengines']['eq_coeffs'] = str([float(eq) for eq in self.equalizer_coeffs])
            
        ### Antenna and IP source
        for i,snap in enumerate(self.config['host']['snaps']):
            sconf['fengines'][snap] = {}
            sconf['fengines'][snap]['ants'] = f"[{i*32}, {(i+1)*32}]"
            sconf['fengines'][snap]['gbe'] = int2ip(ip2int(self.config['snap']['data_ip_base']) + i)
            sconf['fengines'][snap]['source_port'] = self.config['snap']['data_port_base']
            
        ### X-engine MAC addresses
        macs = load_ethers()
        for ip,mac in macs.items():
            sconf['xengines']['arp'][ip] = '0x'+mac.replace(':', '')
            
        ### X-engine channel mapping
        i = 0
        for ip in macs.keys():
            chan0 = self.config['drx'][i]['first_channel']
            nchan = int(round(self.config['drx'][i]['capture_bandwidth'] / CHAN_BW))
            port = self.config['server']['data_ports'][i]
        
            sconf['xengines']['chans'][f"{ip}-{port}"] = f"[{chan0}, {chan0+nchan}]"
            i += 1
            
        ### Save
        sconf = yaml.dump(sconf)
        configname = '/tmp/snap_config.yaml'
        with open(configname, 'w') as fh:
            fh.write(sconf.replace("'", ''))
            
        # Go!
        success = False
        with self.access_lock:
            if self.snap.is_connected and self.snap.fpga.is_programmed():
                for i in range(self.config['snap']['max_program_attempts']):
                    try:
                        self.snap.cold_start_from_config(configname) 
                        sucesss = True
                        break
                    except Exception as e:
                        pass
                        
        return success
        
    def get_spectra(self, t_int=0.1):
        # Return a 2-D array of auto-correlation spectra
        
        acc_len = int(round(CHAN_BW * t_int))
        
        spectra = []
        with self.access_lock:
            if self.snap.is_connected and self.snap.fpga.is_programmed():
                self.snap.autocorr.set_acc_len(acc_len)
                time.sleep(t_int+0.1)
                
                for i in range(4):
                    spectra.append( self.snap.autocorr.get_new_spectra(signal_block=i) )
        spectra = np.array(spectra)
        spectra = spectra.reshape(-1, spectra.shape[-1])
        return spectra
        
    # TODO: Configure channel selection (based on FST)
    # TODO: start/stop data flow (remember to call snap.reset() before start)


def exception_in(vals, error_type=Exception):
    return any([isinstance(val, error_type) for val in vals])


class MsgProcessor(ConsumerThread):
    def __init__(self, config, log,
                max_process_time=1.0, ncmd_save=4, dry_run=False):
        ConsumerThread.__init__(self)
        
        self.config           = config
        self.log              = log
        self.shutdown_timeout = 3.
        self.dry_run          = dry_run
        self.msg_queue        = Queue()
        max_concurrent_msgs = int(MAX_MSGS_PER_SEC*max_process_time)
        self.thread_pool = ThreadPool(max_concurrent_msgs)
        self.name = "Ndp.MsgProcessor"
        self.utc_start     = None
        self.utc_start_str = "NULL"
        
        self.messageServer = ISC.PipelineMessageServer(addr=('ndp',5832))
        
        mcs_local_host  = self.config['mcs']['headnode']['local_host']
        mcs_local_port  = self.config['mcs']['headnode']['local_port']
        mcs_remote_host = self.config['mcs']['headnode']['remote_host']
        mcs_remote_port = self.config['mcs']['headnode']['remote_port']
        """
        self.msg_receiver = MCS2.MsgReceiver((mcs_local_host, mcs_local_port),
                                            subsystem=SUBSYSTEM)
        self.msg_sender   = MCS2.MsgSender((mcs_remote_host, mcs_remote_port),
                                        subsystem=SUBSYSTEM)
        """
        # Maps slot->[(cmd,ref,exit_code), ...]
        self.cmd_status = SequenceDict(list, maxlen=ncmd_save)
        #self.zmqctx = zmq.Context()
        
        self.headnode = ObjectPool([NdpServerMonitorClient(config, log, 'ndp'),])
        self.servers = ObjectPool([NdpServerMonitorClient(config, log, host)
                                for host in self.config['host']['servers']])
        nsnap = NBOARD
        self.snaps = ObjectPool([Snap2MonitorClient(config, log, num+1)
                                for num in range(nsnap)])
        
        #self.fst = Fst(config, log)
        self.drx = Drx(config, log, self.messageServer, self.servers)
        self.tbf = Tbf(config, log, self.messageServer, self.servers)
        self.bam = Bam(config, log, self.messageServer, self.servers)
        self.cor = Cor(config, log, self.messageServer, self.servers)

        self.serial_number = '1'
        self.version = str(__version__)
        self.state = {}
        self.state['status']  = 'SHUTDWN'
        self.state['info']    = 'Need to INI NDP'
        self.state['lastlog'] = ('Welcome to NDP S/N %s, version %s' %
                                (self.serial_number, self.version))
        self.state['activeProcess'] = []
        self.ready = False
        
        self.shutdown_event = threading.Event()
        
        self.run_execute_thread = threading.Thread(target=self.run_execute)
        self.run_execute_thread.daemon = True
        self.run_execute_thread.start()
        
        self.run_monitor_thread = threading.Thread(target=self.run_monitor)
        self.run_monitor_thread.daemon = True
        self.run_monitor_thread.start()
        
        self.run_failsafe_thread = threading.Thread(target=self.run_failsafe)
        self.run_failsafe_thread.daemon = True
        self.run_failsafe_thread.start()
        
        self.start_lock_thread()
        self.start_internal_trigger_thread()
        
    @ISC.logException
    def start_lock_thread(self):
        self.lock_server = ISC.PipelineEventServer(addr=('ndp',5834), timeout=300)
        self.lock_server.start()
        
    @ISC.logException
    def stop_lock_thread(self):
        try:
            self.lock_server.stop()
            del self.lock_server
        except AttributeError:
            pass
            
    def internal_trigger_callback(self, timestamp):
        if os.path.exists(TRIGGERING_ACTIVE_FILE):
            self.log.info('Processing internal trigger at %.6fs', 1.0*timestamp/FS)
            # Wait 1 second to make sure the data is in the buffer
            time.sleep(1.0)
            # Dump 1000 ms of data locally from both tunings, starting 500 ms prior to the trigger
            self.messageServer.trigger(timestamp-98000000, 196000000, 3, local=True)
            
    def start_internal_trigger_thread(self):
        self.internal_trigger_server = ISC.InternalTriggerProcessor(deadtime=200, 
                                                                    callback=self.internal_trigger_callback)
        self.run_internal_trigger_thread = threading.Thread(target=self.internal_trigger_server.run)
        self.run_internal_trigger_thread.start()
        
    def stop_internal_trigger_thread(self):
        try:
            os.unlink(TRIGGERING_ACTIVE_FILE)
        except OSError:
            pass
        self.internal_trigger_server.shutdown()
        self.run_internal_trigger_thread.join()
        del self.internal_trigger_server
        
    def uptime(self):
        # Returns no. secs since data processing began (during INI)
        if self.utc_start is None:
            return 0
        secs = (datetime.datetime.utcnow() - self.utc_start).total_seconds()
        return secs
        
    def raise_error_state(self, cmd, state):
        # TODO: Need new codes? Need to document them?
        state_map = {'BOARD_SHUTDOWN_FAILED':      (0x08,'Board-level shutdown failed'),
                     'BOARD_PROGRAMMING_FAILED':   (0x04,'Board programming failed'),
                     'BOARD_CONFIGURATION_FAILED': (0x05,'Board configuration failed'),
                     'SERVER_STARTUP_FAILED':      (0x09,'Server startup failed'),
                     'SERVER_SHUTDOWN_FAILED':     (0x0A,'Server shutdown failed'), 
                     'PIPELINE_STARTUP_FAILED':    (0x0B,'Pipeline startup failed'),
                     'ADC_CALIBRATION_FAILED':     (0x0C,'ADC offset calibration failed'),
                     'ROACH_FFT_SYNC_FAILED':      (0x0D,'Roach FFT window out of sync'),
                     'PIPLINE_PROCESSING_ERROR':   (0x0E,'Pipeline processing error')}
        code, msg = state_map[state]
        self.state['lastlog'] = '%s: Finished with error' % cmd
        self.state['status']  = 'ERROR'
        self.state['info']    = 'SUMMARY! 0x%02X! %s' % (code, msg)
        self.state['activeProcess'].pop()
        return code
        
    def check_success(self, func, description, names):
        self.state['info'] = description
        rets = func()
        #self.log.info("check_success rets: "+' '.join([str(r) for r in rets]))
        oks = [True for _ in rets]
        for i, (name, ret) in enumerate(zip(names, rets)):
            if isinstance(ret, Exception):
                oks[i] = False
                self.log.error("%s: %s" % (name, str(ret)))
        all_ok = all(oks)
        if not all_ok:
            symbols = ''.join(['.' if ok else 'x' for ok in oks])
            self.log.error("%s failed: %s" % (description, symbols))
            self.state['info'] = description + " failed"
        else:
            self.state['info'] = description + " succeeded"
        return all_ok
        
    def ini(self, arg=None):
        start_time = time.time()
        # Note: Return value from this function is not used
        self.ready = False
        self.state['activeProcess'].append('INI')
        self.state['status'] = 'BOOTING'
        self.state['info']   = 'Running INI sequence'
        self.log.info("Running INI sequence")
        
        # Figure out if the servers are up or not
        if not all(self.servers.can_ssh()):
            ## Down, power them on
            self.log.info("Powering on servers")
            if not self.check_success(lambda: self.servers.do_power('on'),
                                      'Powering on servers',
                                      self.servers.host):
                if 'FORCE' not in arg:
                    return self.raise_error_state('INI', 'SERVER_STARTUP_FAILED')
            startup_timeout = self.config['server']['startup_timeout']
            try:
                #self._wait_until_servers_power('on', startup_timeout)
                # **TODO: Use this instead when Paramiko issues resolved!
                self._wait_until_servers_can_ssh(    startup_timeout)
            except RuntimeError:
                if 'FORCE' not in arg:
                    return self.raise_error_state('INI', 'SERVER_STARTUP_FAILED')
                    
        ## Stop the pipelines
        self.log.info('Stopping pipelines')
        for tuning in range(4):
            self.servers.stop_drx(tuning=tuning)
        for beam in range(4):
            self.headnode.stop_tengine(beam=beam)
            
        ## Make sure the pipelines have stopped
        try:
            self._wait_until_pipelines_stopped(max_wait=40)
        except RuntimeError:
            self.log.warning('Some pipelines have failed to stop, trying harder')
            for beam in range(4):
                for server in self.headnode:
                    pids = server.pid_tengine(beam=beam)
                    for pid in filter(lambda x: x > 0, pids):
                        self.log.warning('  Killing %s TEngine-%i, PID %i', server.host, beam, pid)
                        server.kill_pid(pid)
            for tuning in range(2):
                for server in self.servers:
                    pids = server.pid_drx(tuning=tuning)
                    for pid in filter(lambda x: x > 0, pids):
                        self.log.warning('  Killing %s DRX-%i, PID %i', server.host, tuning, pid)
                        server.kill_pid(pid)
                        
        self.log.info("Forcing CPUs into performance mode")
        self.headnode._shell_command('/root/fixCPU.sh')
        self.servers._shell_command('/root/fixCPU.sh')
        
        self.log.info("Stopping Lock thread")
        self.stop_lock_thread()
        time.sleep(3)
        self.log.info("Starting Lock thread")
        self.start_lock_thread()
        
        self.log.info("Stopping Internal Trigger thread")
        self.stop_internal_trigger_thread()
        time.sleep(3)
        self.log.info("Starting Internal Trigger thread")
        self.start_internal_trigger_thread()
        
        # Note: Must do this to ensure pipelines wait for the new UTC_START
        self.utc_start     = None
        self.utc_start_str = 'NULL'
        
        # Reset the internal state for the various modes/commands 
        self.drx._reset_state()
        self.tbf._reset_state()
        self.bam._reset_state()
        self.cor._reset_state()
        
        # Bring up the pipelines
        can_ssh_status = ''.join(['.' if ok else 'x' for ok in self.servers.can_ssh()])
        self.log.info("Can ssh: "+can_ssh_status)
        if all(self.servers.can_ssh()) or 'FORCE' in arg:
            self.log.info("Restarting pipelines")
            for beam in range(self.config['drx'][0]['beam_count']):
                if not self.check_success(lambda: self.headnode.restart_tengine(beam=beam),
                                          'Restarting pipelines - DRX/T-engine',
                                          self.headnode.host):
                    if 'FORCE' not in arg:
                        return self.raise_error_state('INI', 'SERVER_STARTUP_FAILED')
            for tuning in range(len(self.config['drx'])):
                if not self.check_success(lambda: self.servers.restart_drx(tuning=tuning),
                                          'Restarting pipelines - DRX',
                                          self.servers.host):
                    if 'FORCE' not in arg:
                        return self.raise_error_state('INI', 'SERVER_STARTUP_FAILED')
                        
        # Bring up the FPGAs
        if 'NOREPROGRAM' not in arg: # Note: This is for debugging, not in spec
            self.log.info("Programming FPGAs with '%s'", self.config['snap']['firmware'])
            if not self.check_success(lambda: self.snaps.program(),
                                      'Programming FPGAs',
                                      self.snaps.host):
                if 'FORCE' not in arg: # Note: Also not in spec
                    return self.raise_error_state('INI', 'BOARD_PROGRAMMING_FAILED')
                    
        self.log.info("Configuring FPGAs")
        if not self.check_success(lambda: [s.configure() for i,s in enumerate(self.snaps) if i == 0],
                                  'Configuring master FPGA',
                                  [s.host for i,s in enumerate(self.snaps) if i == 0]):
            if 'FORCE' not in arg:
                return self.raise_error_state('INI', 'BOARD_CONFIGURATION_FAILED')
        if not self.check_success(lambda: [s.configure() for i,s in enumerate(self.snaps) if i != 0],
                                  'Configuring remaining FPGAs',
                                  [s.host for i,s in enumerate(self.snaps) if i != 0]):
            if 'FORCE' not in arg:
                return self.raise_error_state('INI', 'BOARD_CONFIGURATION_FAILED')
                
        self.log.info("  Finished configuring FPGAs")
        
        self.utc_start     = self.snaps[0].get_tt_of_sync()
        self.utc_start_str = str(self.utc_start)
        self.state['lastlog'] = "Starting correlator processing at timetag "+self.utc_start_str
        self.log.info("Starting correlator processing at timetag "+self.utc_start_str)
        time.sleep(0.5)
        
        # Check and make sure that *all* of the pipelines started
        self.log.info("Checking pipeline processing")
        ## DRX
        pipeline_pids = []
        for tuning in range(len(self.config['drx'])):
            pipeline_pids = [p for s in self.servers.pid_drx(tuning=tuning) for p in s]
            pipeline_pids = list(filter(lambda x: x>0, pipeline_pids))
            print('DRX-%i:' % tuning, len(pipeline_pids), pipeline_pids)
            if len(pipeline_pids) != 1:
                self.log.error('Found %i DRX-%i pipelines running, expected %i', len(pipeline_pids), tuning, 1)
                if 'FORCE' not in arg:
                    return self.raise_error_state('INI', 'PIPELINE_STARTUP_FAILED')
        ## T-engine
        for beam in range(self.config['drx'][0]['beam_count']):
            pipeline_pids = [p for s in self.headnode.pid_tengine(beam=beam) for p in s]
            pipeline_pids = list(filter(lambda x: x>0, pipeline_pids))
            print('TEngine-%i:' % beam, len(pipeline_pids), pipeline_pids)
            if len(pipeline_pids) != 1:
                self.log.error('Found %i TEngine-%i pipelines running, expected %i', len(pipeline_pids), beam,  1)
                if 'FORCE' not in arg:
                    return self.raise_error_state('INI', 'PIPELINE_STARTUP_FAILED')
        self.log.info('Checking pipeline processing succeeded')
        
        #self.log.info("Starting correlator")
        # TODO: Should we do something here?
        
        self.log.info("INI finished in %.3f s", time.time() - start_time)
        self.ready = True
        self.state['lastlog'] = 'INI finished in %.3f s' % (time.time() - start_time)
        self.state['status']  = 'NORMAL'
        self.state['info'] = 'System calibrated and operating normally'
        self.state['activeProcess'].pop()
        return 0
        
    def sht(self, arg=''):
        # TODO: Consider allowing specification of 'only servers' or 'only boards'
        start_time = time.time()
        self.ready = False
        self.state['activeProcess'].append('SHT')
        self.state['status'] = 'SHUTDWN'
        # TODO: Use self.check_success here like in ini()
        self.log.info("System is shutting down")
        self.state['info']   = 'System is shutting down'
        do_reboot = ('HARD' in arg)
        if 'SCRAM' in arg:
            if 'RESTART' in arg:
                if exception_in(self.servers.do_power('reset')):
                    if 'FORCE' not in arg:
                        return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
                if exception_in(self.snaps.unprogram(do_reboot)):
                    if 'FORCE' not in arg:
                        return self.raise_error_state('SHT', 'BOARD_SHUTDOWN_FAILED')
            else:
                if exception_in(self.servers.do_power('off')):
                    if 'FORCE' not in arg:
                        return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
                if exception_in(self.snaps.unprogram(do_reboot)):
                    if 'FORCE' not in arg:
                        return self.raise_error_state('SHT', 'BOARD_SHUTDOWN_FAILED')
            self.log.info("SHT SCRAM finished in %.3f s", time.time() - start_time)
            self.state['lastlog'] = 'SHT finished in %.3f s' % (time.time() - start_time)
            self.state['status']  = 'SHUTDWN'
            self.state['info']    = 'System has been shut down'
            self.state['activeProcess'].pop()
        else:
            if 'RESTART' in arg:
                def soft_reboot():
                    self.log.info('Shutting down servers')
                    if exception_in(self.servers.do_power('soft')):
                        if 'FORCE' not in arg:
                            return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
                    self.log.info('Unprogramming snaps')
                    if exception_in(self.snaps.unprogram(do_reboot)):
                        if 'FORCE' not in arg:
                            return self.raise_error_state('SHT', 'BOARD_SHUTDOWN_FAILED')
                    self.log.info('Waiting for servers to power off')
                    try:
                        self._wait_until_servers_power('off', max_wait=180)
                    except RuntimeError:
                        if 'FORCE' not in arg:
                            return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
                    self.log.info('Powering on servers')
                    if exception_in(self.servers.do_power('on')):
                        if 'FORCE' not in arg:
                            return self.raise_error_state('SHT', 'SERVER_STARTUP_FAILED')
                    self.log.info('Waiting for servers to power on')
                    try:
                        self._wait_until_servers_power('on')
                    except RuntimeError:
                        if 'FORCE' not in arg:
                            return self.raise_error_state('SHT', 'SERVER_STARTUP_FAILED')
                    self.log.info("SHT RESTART finished in %.3f s", time.time() - start_time)
                    self.state['lastlog'] = 'SHT finished in %.3f s' % (time.time() - start_time)
                    self.state['status']  = 'SHUTDWN'
                    self.state['info']    = 'System has been shut down'
                    self.state['activeProcess'].pop()
                self.thread_pool.add_task(soft_reboot)
            else:
                def soft_power_off():
                    self.log.info('Shutting down servers')
                    if exception_in(self.servers.do_power('soft')):
                        if 'FORCE' not in arg:
                            return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
                    self.log.info('Unprogramming snaps')
                    if exception_in(self.snaps.unprogram(do_reboot)):
                        if 'FORCE' not in arg:
                            return self.raise_error_state('SHT', 'BOARD_SHUTDOWN_FAILED')
                    self.log.info('Waiting for servers to power off')
                    try:
                        self._wait_until_servers_power('off', max_wait=180)
                    except RuntimeError:
                        if 'FORCE' not in arg:
                            return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
                    self.log.info("SHT finished in %.3f s", time.time() - start_time)
                    self.state['lastlog'] = 'SHT finished in %.3f s' % (time.time() - start_time)
                    self.state['status']  = 'SHUTDWN'
                    self.state['info']    = 'System has been shut down'
                    self.state['activeProcess'].pop()
                self.thread_pool.add_task(soft_power_off)
        return 0
        
    def _wait_until_servers_power(self, target_state, max_wait=60):
        # TODO: Need to check for ping (or even ssh connectivity) instead of 'power is on'?
        time.sleep(6)
        wait_time = 6
        while not all( (state == target_state
                        for state in self.servers.get_power_state()) ):
            time.sleep(2)
            wait_time += 2
            if wait_time >= max_wait:
                raise RuntimeError("Timed out waiting for server(s) to turn "+target_state)
                
    def _wait_until_servers_can_ssh(self, max_wait=60):
        wait_time = 0
        while not all(self.servers.can_ssh()):
            time.sleep(2)
            wait_time += 2
            if wait_time >= max_wait:
                raise RuntimeError("Timed out waiting to ssh to server(s)")
                
    def _wait_until_pipelines_stopped(self, max_wait=60):
        nRunning = 1
        t0, t1 = time.time(), time.time()
        while nRunning > 0:
            pids = []
            for tuning in range(4):
                for server in self.servers:
                    pids.extend( server.pid_drx(tuning=tuning) )
            for beam in range(4):
                for server in self.headnode:
                    pids.extend( server.pid_tengine(beam=beam) )
            nRunning = len( list(filter(lambda x: x > 0, pids)) )
            
            t1 = time.time()
            if t1-t0 >= max_wait:
                raise RuntimeError("Timed out waiting for pipelines to stop")
            time.sleep(5)
            
    def run_execute(self):
        self.log.info("Starting slot execution thread")
        slot = MCS2.get_current_slot()
        while not self.shutdown_event.is_set():
            for cmd_processor in [self.drx, self.tbf, self.bam, self.cor]:#, self.fst]
                self.thread_pool.add_task(cmd_processor.execute_commands,
                                        slot)
            while MCS2.get_current_slot() == slot:
                time.sleep(0.1)
            time.sleep(0.1)
            slot += 1
            
    def run_monitor(self):
        self.log.info("Starting monitor thread")
        
        # Assumes that we are running on the headnode, which should always be true
        pipelines = OrderedDict()
        pipelines['localhost'] = BifrostPipelines('localhost').pipelines()
        for server in self.servers:
            host = server.host.replace('-data', '')
            pipelines[host] = BifrostPipelines(host).pipelines()
            
        # Needed to figure out when to ignore the T-engine output
        tbf_lock = ISC.PipelineEventClient(addr=('ndp',5834))
        
        # A little state to see if we need to re-check hosts
        force_recheck = False if self.ready else True
        
        # Go!
        n_beams = self.config['drx'][0]['beam_count']
        n_tunings = len(self.config['drx'])
        n_servers = len(self.config['host']['servers-data'])
        while not self.shutdown_event.is_set():
            ## A little more state
            problems_found = False
            
            if self.ready:
                ## Check the servers
                found = {'drx':[], 'tengine':[]}
                for host in list(pipelines.keys()):
                    ### Basic information about what to expect
                    n_expected = n_beams if host == 'localhost' else n_tunings
                    
                    ### Check to see if our view of which pipelines are running has changed
                    refresh = False if len(pipelines[host]) == n_expected else True
                    for pipeline in pipelines[host]:
                        if not pipeline.is_alive():
                            refresh = True
                            break
                    if refresh or force_recheck:
                        del pipelines[host]
                        pipelines[host] = BifrostPipelines(host).pipelines()
                        
                    ### Loop over the pipelines
                    for pipeline in pipelines[host]:
                        name = pipeline.command
                        side = 0
                        if name.find('--tuning 1') != -1 or name.find('--beam 1') != -1:
                            side = 1
                        elif name.find('--tuning 2') != -1 or name.find('--beam 2') != -1:
                            side = 2
                        elif name.find('--tuning 3') != -1 or name.find('--beam 3') != -1:
                            side = 3
                        loss = pipeline.rx_loss()
                        txbw = pipeline.tx_rate()
                        cact = pipeline.is_corr_active()
                        
                        if name.find('drx') != -1:
                            found['drx'].append( (host,name,side,loss,txbw,cact) )
                        elif name.find('tengine') != -1:
                            found['tengine'].append( (host,name,side,loss,txbw) )
                        else:
                            pass
                            
                ## Make sure we have everything we need
                ### T-engines
                if not self.ready:
                    ## Deal with the system shutting down in the middle of a poll
                    continue
                total_tengine_bw = {0:0, 1:0, 2:0, 3:0}
                for host,name,side,loss,txbw in found['tengine']:
                    total_tengine_bw[side] += txbw
                    if loss > 0.01:    # >1% packet loss
                        problems_found = True
                        msg = "%s, T-Engine-%i -- RX loss of %.1f%%" % (host, side, loss*100.0)
                        if self.state['status'] != 'ERROR':
                            self.state['lastlog'] = msg
                            self.state['status'] = 'WARNING'
                            self.state['info']    = '%s! 0x%02X! %s' % ('SUMMARY', 0x0E, msg)
                        self.log.warning(msg)
                for side in range(n_beam):
                    if self.drx.cur_freq[side*2] > 0 and not tbf_lock.is_set() and total_tengine_bw[side*2] == 0:
                        problems_found = True
                        msg = "T-Engine-%i -- TX rate of %.1f MB/s" % (side, total_tengine_bw[side]/1024.0**2)
                        self.state['lastlog'] = msg
                        self.state['status']  = 'ERROR'
                        self.state['info']    = '%s! 0x%02X! %s' % ('SUMMARY', 0x0E, msg)
                        self.log.error(msg)
                if len(found['tengine']) != n_beams:
                    problems_found = True
                    msg = "Found %i T-Engines instead of %i" % (len(found['tengine']), n_beams)
                    self.state['lastlog'] = msg
                    self.state['status']  = 'ERROR'
                    self.state['info']    = '%s! 0x%02X! %s' % ('SUMMARY', 0x0E, msg)
                    self.log.error(msg)
                    
                ### DRX pipelines
                if not self.ready:
                    ## Deal with the system shutting down in the middle of a poll
                    continue
                total_drx_bw = {0:0, 1:0, 2:0, 3:0}
                total_drx_inactive = {0:0, 1:0, 2:0, 3:0}
                for host,name,side,loss,txbw,cact in found['drx']:
                    total_drx_bw[side] += txbw
                    total_drx_inactive[side] += (1 if txbw == 0 else 0)
                    if loss > 0.01:    # >1% packet loss
                        problems_found = True
                        msg = "%s, DRX-%i -- RX loss of %.1f%%" % (host, side, loss*100.0)
                        if self.state['status'] != 'ERROR':
                            self.state['lastlog'] = msg
                            self.state['status'] = 'WARNING'
                            self.state['info']    = '%s! 0x%02X! %s' % ('SUMMARY', 0x0E, msg)
                        self.log.warning(msg)
                    if self.drx.cur_freq[side] > 0 and side == 0 and not cact:
                        problems_found = True
                        msg = "%s, DRX-%i -- Correlator not running" % (host, side)
                        self.state['lastlog'] = msg
                        self.state['status']  = 'ERROR'
                        self.state['info']    = '%s! 0x%02X! %s' % ('SUMMARY', 0x0E, msg)
                        self.log.error(msg)
                for side in range(n_tunings):
                    if self.drx.cur_freq[side] > 0 and total_drx_inactive[side] > 0:
                        problems_found = True
                        msg = "DRX-%i -- TX rate of %.1f MB/s" % (side, total_drx_bw[side]/1024.0**2)
                        self.state['lastlog'] = msg
                        self.state['status']  = 'ERROR'
                        self.state['info']    = '%s! 0x%02X! %s' % ('SUMMARY', 0x0E, msg)
                        self.log.error(msg)
                if len(found['drx']) != n_tunings*n_servers:
                    problems_found = True
                    msg = "Found %i DRX pipelines instead of %i" % (len(found['drx']), n_tunings*n_servers)
                    if self.state['status'] != 'ERROR':
                        self.state['lastlog'] = msg
                        self.state['status']  = 'WARNING'
                        self.state['info']    = '%s! 0x%02X! %s' % ('SUMMARY', 0x0E, msg)
                    self.log.warning(msg)
                    
                ## Check the snap boards
                if not self.ready:
                    ## Deal with the system shutting down in the middle of a poll
                    continue
                snaps_programmed = self.snaps.is_programmed()
                if not all(snaps_programmed):
                    problems_found = True
                    msg = "Found %s SNAP2 board(s) not programmed" % (len(snaps_programmed) - sum(snaps_programmed),)
                    self.state['lastlog'] = msg
                    self.state['status']  = 'ERROR'
                    self.state['info']    = '%s! 0x%02X! %s' % ('SUMMARY', 0x0E, msg)
                    self.log.error(msg)
                    
                ## De-assert anything that we can de-assert
                if not self.ready:
                    ## Deal with the system shutting down in the middle of a poll
                    continue
                if not problems_found:
                    if self.state['status'] == 'WARNING':
                        msg = 'Warning condition(s) cleared'
                        self.state['lastlog'] = msg
                        self.state['status']  = 'NORMAL'
                        self.state['info']    = msg
                        self.log.info(msg)
                    elif self.state['status'] == 'ERROR' and self.state['info'].find('0x0E') != -1:
                        msg = 'Pipeline error condition(s) cleared, dropping back to warning'
                        self.state['lastlog'] = msg
                        self.state['status']  = 'WARNING'
                        self.state['info']    = '%s! 0x%02X! %s' % ('WARNING', 0x0E, msg)
                        self.log.info(msg)
                force_recheck = False
                
                self.log.info("Monitor OK")
                time.sleep(self.config['monitor_interval'])
                
            else:
                force_recheck = True
                
                self.log.info("Monitor SKIP")
                time.sleep(30)
                
    def run_failsafe(self):
        self.log.info("Starting failsafe thread")
        while not self.shutdown_event.is_set():
            slot = MCS2.get_current_slot()
            
            # Note: Actually just flattening lists, not summing
            server_temps = sum([list(v) for v in self.servers.get_temperatures(slot).values()], [])
            # Remove error values before reducing
            server_temps = [val for val in server_temps if not math.isnan(val)]
            if len(server_temps) == 0: # If all values were nan (exceptional!)
                server_temps = [float('nan')]
            server_temps_max = np.max(server_temps)
            if server_temps_max > self.config['server']['temperature_shutdown']:
                self.state['lastlog'] = 'Temperature shutdown -- server'
                self.state['status']  = 'ERROR'
                self.state['info']    = '%s! 0x%02X! %s' % ('SERVER_TEMP_MAX', 0x01,
                                                            'Server temperature shutdown')
                if server_temps_max > self.config['server']['temperature_scram']:
                    self.sht('SCRAM')
                else:
                    self.sht()
            elif server_temps_max > self.config['server']['temperature_warning']:
                if self.state['status'] != 'ERROR':
                    self.state['lastlog'] = 'Temperature warning -- server'
                    self.state['status']  = 'WARNING'
                    self.state['info']    = '%s! 0x%02X! %s' % ('SERVER_TEMP_MAX', 0x01,
                                                                'Server temperature warning')
                    
            # Note: Actually just flattening lists, not summing
            snap_temps  = sum([list(v) for v in self.snaps.get_temperatures(slot).values()], [])
            # Remove error values before reducing
            snap_temps  = [val for val in snap_temps  if not math.isnan(val)]
            if len(snap_temps) == 0: # If all values were nan (exceptional!)
                snap_temps = [float('nan')]
            snap_temps_max  = np.max(snap_temps)
            if snap_temps_max > self.config['snap']['temperature_shutdown']:
                self.state['lastlog'] = 'Temperature shutdown -- snap'
                self.state['status']  = 'ERROR'
                self.state['info']    = '%s! 0x%02X! %s' % ('BOARD_TEMP_MAX', 0x01,
                                                            'Board temperature shutdown')
                if snap_temps_max > self.config['snap']['temperature_scram']:
                    self.sht('SCRAM')
                else:
                    self.sht()
            elif snap_temps_max > self.config['snap']['temperature_warning']:
                if self.status['status'] != 'ERROR':
                    self.state['lastlog'] = 'Temperature warning -- snap'
                    self.state['status']  = 'WARNING'
                    self.state['info']    = '%s! 0x%02X! %s' % ('BOARD_TEMP_MAX', 0x01,
                                                                'Board temperature warning')
                    
            self.log.info("Failsafe OK")
            time.sleep(self.config['failsafe_interval'])
            
    def process(self, msg):
        if msg.cmd == 'PNG':
            self.log.info('Received PNG: '+str(msg))
            if not self.dry_run:
                self.process_msg(msg, lambda msg: True, '')
        elif msg.cmd == 'RPT':
            if msg.data != 'UTC_START':
                self.log.info('Received RPT request: '+str(msg))
            if not self.dry_run:
                # Note: RPT messages are processed asynchronously
                #         to avoid stalls.
                # TODO: Check that this doesn't cause any problems
                #         due to race conditions.
                self.thread_pool.add_task(self.process_msg,
                                          msg, self.process_report)
        else:
            self.log.info('Received command: '+str(msg))
            if not self.dry_run:
                self.process_msg(msg, self.process_command)
                
    def shutdown(self):
        self.shutdown_event.set()
        self.stop_synchronizer_thread()
        self.stop_lock_thread()
        self.stop_internal_trigger_thread()
        # Propagate shutdown to downstream consumers
        self.msg_queue.put(ConsumerThread.STOP)
        if not self.thread_pool.wait(self.shutdown_timeout):
            self.log.warning("Active tasks still exist and will be killed")
        self.run_execute_thread.join(self.shutdown_timeout)
        if self.run_execute_thread.isAlive():
            self.log.warning("run_execute thread still exists and will be killed")
        self.run_monitor_thread.join(self.shutdown_timeout)
        if self.run_monitor_thread.isAlive():
            self.log.warning("run_monitor thread still exists and will be killed")
        print(self.name, "shutdown")
        
    def process_msg(self, msg, process_func):
        accept, reply_data = process_func(msg)
        status = self.state['status']
        reply_msg = msg.create_reply(accept, status, reply_data)
        self.msg_queue.put(reply_msg)
        
    def process_report(self, msg):
        key, args = MCS2.mib_parse_label(msg.data)
        try: value = self._get_report_result(key, args, msg.slot)
        except KeyError:
            self.log.warning('Unknown MIB entry: %s' % msg.data)
            return False, 'Unknown MIB entry: %s' % msg.data
        except ValueError as e:
            self.log.warning(e)
            return False, str(e)
        #except (ValueError,RuntimeError) as e:
        except Exception as e:
            self.log.error('%s: %s'%(type(e), str(e)))
            return False, '%s: %s'%(type(e), str(e))
        reply_data = self._pack_report_result(key, value)
        log_data   = self._format_report_result(key, value)
        self.log.debug('%s = %s' % (msg.data, log_data))
        return True, reply_data
        
    def _get_next_fir_index(self):
        idx = self.fir_idx
        self.fir_idx += 1
        self.fir_idx %= NINPUT
        return idx
        
    def _get_snap_config(self):
        sub_config = {}
        for key in ('firmware', 'fft_shift', 'equalizer_coeffs', 'bypass_pfb'):
            sub_config[key] = self.config['snap'][key]
        sub_config['equalizer_coeffs_md5'] = ''
        
        try:
            m = hashlib.md5()
            with open(sub_config['equalizer_coeffs'], 'rb') as fh:
                m.update(fh.read())
            sub_config['equalizer_coeffs_md5'] = str(m.hexdigest())
        except (OSError, IOError):
            pass
            
        return json.dumps(sub_config)
        
    def _get_tengine_config(self):
        sub_config = {}
        sub_config['nbeam'] = self.config['drx'][0]['beam_count']
        for i,engine in enumerate(self.config['tengine']):
            sub_config['pfb_inverter'+str(i)] = engine['pfb_inverter']
            
        return json.dumps(sub_config)
        
    def _get_report_result(self, key, args, slot):
        reduce_ops = {'MAX':      np.max,
                      'MIN':      np.min,
                      'AVG':      np.mean,
                      'RMS':      lambda x: np.sqrt(np.mean(x**2)),
                      'SAT':      lambda x: np.sum(np.abs(x)>=ADC_MAXVAL),
                      'DCOFFSET': np.mean,
                      'PEAK':     np.max}
        if key == 'SUMMARY':           return self.state['status']
        if key == 'INFO':              return self.state['info']
        if key == 'LASTLOG':           return self.state['lastlog']
        if key == 'SUBSYSTEM':         return SUBSYSTEM
        if key == 'SERIALNO':          return self.serial_number
        if key == 'VERSION':           return self.version
        if key == 'SNAP_CONFIG':       return self._get_snap_config()
        if key == 'TENGINE_CONFIG':    return self._get_tengine_config()
        # TODO: TBF_STATUS
        #       TBF_TUNING_MASK
        if key == 'NUM_STANDS':        return NSTAND
        if key == 'NUM_SERVERS':       return NSERVER
        if key == 'NUM_BOARDS':        return NBOARD
        # TODO: NUM_BEAMS
        if key == 'BEAM_FIR_COEFFS':   return FIR_NCOEF
        # TODO: T_NOM
        if key == 'NUM_DRX_TUNINGS':   return self.drx.ntuning
        if args[0] == 'DRX' and args[1] == 'CONFIG':
            tuning = args[2]-1
            if args[3] == 'FREQ':
                return self.drx.cur_freq[tuning]
            if args[3] == 'FILTER':
                return self.drx.cur_filt[tuning]
            if args[3] == 'GAIN':
                return self.drx.cur_gain[tuning]
        if key == 'NUM_FREQ_CHANS':    return NCHAN
        if key == 'FIR_CHAN_INDEX':    return self._get_next_fir_index()
        if key == 'FIR':
            return self.fst.get_fir_coefs(slot)[input2standpol(self.fir_idx)]
        if key == 'CLK_VAL':           return MCS2.slot2mpm(slot-1)
        if key == 'UTC_START':         return self.utc_start_str # Not in spec
        if key == 'UPTIME':            return self.uptime() # Not in spec
        if key == 'STAT_SAMP_SIZE':    return STAT_SAMP_SIZE
        if args[0] == 'ANT':
            inp = args[1]-1
            if not (0 <= inp < NINPUT):
                raise ValueError("Unknown input number %i"%(inp+1))
            board,stand,pol = input2boardstandpol(inp)
            samples = self.snaps[board].get_samples(slot, stand, pol,
                                                    STAT_SAMP_SIZE)
            # Convert from int8 --> float32 before reducing
            samples = samples.astype(np.float32)
            op = args[2]
            return reduce_ops[op](samples)
        # TODO: BEAM_*
        #  BEAM%i_DELAY
        #  BEAM%i_GAIN
        #  BEAM%i_TUNING # Note: (NDP only)
        if args[0] == 'HEALTH' and args[1] == 'CHECK':
            t_now = datetime.datetime.utcnow()
            spectra = self.snaps.get_spectra()
            spectra = np.array(list(spectra))
            spectra = spectra.reshape(-1, spectra.shape[-1])
            spectra = spectra.astype(np.float32)
            
            checkname = t_now.strftime("%y%m%d_%H%M%S")
            checkname = "/tmp/"+checkname+"_snapspecs.dat"
            with open(checkname, 'wb') as fh:
                fh.write(struct.pack('ll', spectra.shape[0]//2, spectra.shape[1]))
                spectra.tofile(fh)
            return checkname
        if args[0] == 'BOARD':
            board = args[1]-1
            if not (0 <= board < NBOARD):
                raise ValueError("Unknown board number %i"%(board+1))
            if args[2] == 'STAT': return None # TODO
            if args[2] == 'INFO': return None # TODO
            if args[2] == 'TEMP':
                temps = self.snaps[board].get_temperatures(slot).values()
                op = args[3]
                return reduce_ops[op](temps)
            if args[2] == 'FIRMWARE': return self.config['snap']['firmware']
            if args[2] == 'HOSTNAME': return self.snaps[board].host
            raise KeyError
        if args[0] == 'SERVER':
            svr = args[1]-1
            if not (0 <= svr < NSERVER):
                raise ValueError("Unknown server number %i"%(svr+1))
            if args[2] == 'HOSTNAME': return self.servers[svr].host
            # TODO: This request() should raise exceptions on failure
            # TODO: Change to .status(), .info()?
            if args[2] == 'STAT': return self.servers[svr].get_status()
            if args[2] == 'INFO': return self.servers[svr].get_info()
            if args[2] == 'TEMP':
                temps = self.servers[svr].get_temperatures(slot).values()
                op = args[3]
                return reduce_ops[op](temps)
            raise KeyError
        if args[0] == 'GLOBAL':
            if args[1] == 'TEMP':
                temps = []
                # Note: Actually just flattening lists, not summing
                temps += sum(self.snaps.get_temperatures(slot).values(), [])
                temps += sum(self.servers.get_temperatures(slot).values(), [])
                # Remove error values before reducing
                temps = [val for val in temps if not math.isnan(val)]
                if len(temps) == 0: # If all values were nan (exceptional!)
                    temps = [float('nan')]
                op = args[2]
                return reduce_ops[op](temps)
            raise KeyError
        if key == 'CMD_STAT': return (slot,self.cmd_status[slot-1])
        raise KeyError
        
    def _pack_report_result(self, key, value):
        return {
            'SUMMARY':            lambda x: x[:7],
            'INFO':               lambda x: truncate_message(x, 256),
            'LASTLOG':            lambda x: truncate_message(x, 256),
            'SUBSYSTEM':          lambda x: x[:3],
            'SERIALNO':           lambda x: x[:5],
            'VERSION':            lambda x: truncate_message(x, 256),
            'ROACH_CONFIG':       lambda x: truncate_message(x, 1024),
            'TENGINE_CONFIG':     lambda x: truncate_message(x, 1024),
            #'TBF_STATUS':
            #'TBF_TUNING_MASK':
            'NUM_DRX_TUNINGS':    lambda x: struct.pack('>B', x),
            'NUM_FREQ_CHANS':     lambda x: struct.pack('>H', x),
            #'NUM_BEAMS':
            'NUM_STANDS':         lambda x: struct.pack('>H', x),
            'NUM_BOARDS':         lambda x: struct.pack('>B', x),
            'NUM_SERVERS':        lambda x: struct.pack('>B', x),
            'BEAM_FIR_COEFFS':    lambda x: struct.pack('>B', x),
            #'T_NOMn:
            'FIR_CHAN_INDEX':     lambda x: struct.pack('>H', x),
            'FIR':                lambda x: x.astype('>h').tobytes(),
            'CLK_VAL':            lambda x: struct.pack('>I', x),
            'UTC_START':          lambda x: truncate_message(x, 256), # Not in spec
            'UPTIME':             lambda x: struct.pack('>I', x),     # Not in spec
            'STAT_SAMPLE_SIZE':   lambda x: struct.pack('>I', x),
            'ANT_RMS':            lambda x: struct.pack('>f', x),
            'ANT_SAT':            lambda x: struct.pack('>i', x),
            'ANT_DCOFFSET':       lambda x: struct.pack('>f', x),
            'ANT_PEAK':           lambda x: struct.pack('>i', x),
            # TODO: Implement these BEAM requests
            #         Are these actually in the spec?
            #'BEAM_RMS':           lambda x: struct.pack('>f', x),
            #'BEAM_SAT':           lambda x: struct.pack('>i', x),
            #'BEAM_DCOFFSET':      lambda x: struct.pack('>f', x),
            #'BEAM_PEAK':          lambda x: struct.pack('>i', x),
            # TODO: In the spec this is >I ?
            'HEALTH_CHECK':       lambda x: truncate_message(x, 1024),
            'BOARD_STAT':         lambda x: struct.pack('>L', x),
            'BOARD_TEMP_MAX':     lambda x: struct.pack('>f', x),
            'BOARD_TEMP_MIN':     lambda x: struct.pack('>f', x),
            'BOARD_TEMP_AVG':     lambda x: struct.pack('>f', x),
            'BOARD_FIRMWARE':     lambda x: truncate_message(x, 256),
            'BOARD_HOSTNAME':     lambda x: truncate_message(x, 256),
            # TODO: SERVER_STAT
            'SERVER_TEMP_MAX':    lambda x: struct.pack('>f', x),
            'SERVER_TEMP_MIN':    lambda x: struct.pack('>f', x),
            'SERVER_TEMP_AVG':    lambda x: struct.pack('>f', x),
            'SERVER_SOFTWARE':    lambda x: truncate_message(x, 256),
            'SERVER_HOSTNAME':    lambda x: truncate_message(x, 256),
            'GLOBAL_TEMP_MAX':    lambda x: struct.pack('>f', x),
            'GLOBAL_TEMP_MIN':    lambda x: struct.pack('>f', x),
            'GLOBAL_TEMP_AVG':    lambda x: struct.pack('>f', x),
            'CMD_STAT':           lambda x: pack_reply_CMD_STAT(*x),
            'DRX_CONFIG_FREQ':  lambda x: struct.pack('>f', x),
            'DRX_CONFIG_FILTER':lambda x: struct.pack('>H', x),
            'DRX_CONFIG_GAIN':  lambda x: struct.pack('>H', x)
        }[key](value)
        
    def _format_report_result(self, key, value):
        format_function = defaultdict(lambda : str)
        format_function.update({
            'FIR':      pretty_print_bytes,
            'CMD_STAT': lambda x: '%i commands in previous slot' % len(x)
        })
        return format_function[key](value)
    
    def currently_processing(self, *cmds):
        return any([cmd in self.state['activeProcess'] for cmd in cmds])
        
    def process_command(self, msg):
        exec_delay = 2
        exec_slot  = msg.slot + exec_delay
        accept = True
        reply_data = ""
        exit_status = 0
        if msg.cmd == 'INI':
            if self.currently_processing('INI', 'SHT'):
                # TODO: This stuff could be tidied up a bit
                self.state['lastlog'] = ('INI: %s - %s is active and blocking'%
                                        ('Blocking operation in progress',
                                        self.state['activeProcess']))
                exit_status = 0x0C
            else:
                self.thread_pool.add_task(self.ini, msg.data)
        elif msg.cmd == 'SHT':
            if self.currently_processing('INI', 'SHT'):
                self.state['lastlog'] = ('SHT: %s - %s is active and blocking'%
                                        ('Blocking operation in progress',
                                        self.state['activeProcess']))
                exit_status = 0x0C
            else:
                self.thread_pool.add_task(self.sht, msg.data)
        elif msg.cmd == 'STP':
            mode = msg.data # TBN/TBF/BEAMn/COR
            if mode == 'DRX':
                # TODO: This is not actually part of the spec (useful for debugging?)
                if self.state['status'] not in ('SHUTDWN', 'BOOTING'):
                    exit_status = self.drx.stop()
                else:
                    self.state['lastlog'] = "STP: Subsystem is not ready"
                    exit_status = 99
            elif mode == 'TBF':
                if self.state['status'] not in ('SHUTDWN', 'BOOTING'):
                    self.state['lastlog'] = "UNIMPLEMENTED STP request"
                    exit_status = -1 # TODO: Implement this
                else:
                    self.state['lastlog'] = "STP: Subsystem is not ready"
                    exit_status = 99
            elif mode.startswith('BEAM'):
                self.state['lastlog'] = "UNIMPLEMENTED STP request"
                exit_status = -1 # TODO: Implement this
                ## Get the beam
                #beam = int(mode[4:], 10)
                ## Build a dummy BAM command that is all zeros for delay/gain on request beam
                #msg.data = struct.pack('>H', beam)
                #msg.data += '\x00'*(1024+2048+2)
                ## Set tuning 1 and send
                #msg.data[-2] = struct.pack('>B', 1)
                #exit_status = self.bam.process_command(msg)
                ## Change to tuning 2 and send again
                #msg.data[-2] = struct.pack('>B', 2)
                #exit_status |= self.bam.process_command(msg)
            elif mode == 'COR':
                self.state['lastlog'] = "UNIMPLEMENTED STP request"
                exit_status = -1 # TODO: Implement this
            else:
                self.state['lastlog'] = "Invalid STP request"
                exit_status = -1
        elif msg.cmd == 'DRX':
            if self.state['status'] not in ('SHUTDWN', 'BOOTING'):
                exit_status = self.drx.process_command(msg)
            else:
                self.state['lastlog'] = "DRX: Subsystem is not ready"
                exit_status = 99
        elif msg.cmd == 'TBF':
            if self.state['status'] not in ('SHUTDWN', 'BOOTING'):
                exit_status = self.tbf.process_command(msg)
            else:
                self.state['lastlog'] = "TBF: Subsystem is not ready"
                exit_status = 99
        elif msg.cmd == 'BAM':
            if self.state['status'] not in ('SHUTDWN', 'BOOTING'):
                exit_status = self.bam.process_command(msg)
            else:
                self.state['lastlog'] = "BAM: Subsystem is not ready"
                exit_status = 99
        elif msg.cmd == 'COR':
            if self.state['status'] not in ('SHUTDWN', 'BOOTING'):
                exit_status = self.cor.process_command(msg)
            else:
                self.state['lastlog'] = "COR: Subsystem is not ready"
                exit_status = 99
        else:
            exit_status = 0
            accept = False
            reply_data = 'Unknown command: %s' % msg.cmd
        if exit_status != 0:
            accept = False
            reply_data = "0x%02X! %s" % (exit_status, self.state['lastlog'])
        self.cmd_status[msg.slot].append( (msg.cmd, msg.ref, exit_status) )
        return accept, reply_data
